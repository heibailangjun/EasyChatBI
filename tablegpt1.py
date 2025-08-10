# tablegpt1.py - 使用Qwen Max的FastAPI后端
import os
import platform
import uuid
import pandas as pd
import json
import re
from io import BytesIO
import base64
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # 兼容OpenAI API
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.staticfiles import StaticFiles
import warnings
from PIL import Image, ImageFilter
import traceback

# --- 1. 屏蔽Pillow警告 ---
warnings.filterwarnings("ignore", category=UserWarning, message="Can't find filter element")

# --- 2. 修复Pillow滤镜 ---
try:
    # 手动注册JPEG必需滤镜
    if not hasattr(ImageFilter, 'DCT'):
        ImageFilter.DCT = lambda mode: ImageFilter.BuiltinFilter("DCT", (3, 3), None, mode)
    if not hasattr(ImageFilter, 'Quantization'):
        ImageFilter.Quantization = lambda: ImageFilter.BuiltinFilter("Quantization", (3, 3), None)
except Exception as e:
    print(f"⚠️ 滤镜修复跳过: {e}")


# 初始化FastAPI应用
app = FastAPI(title="TableGPT Analyzer with Qwen Max")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 文件存储目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 字体设置（matplot出图需正确显示中文字体）
system_os = platform.system()

if system_os == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # Win 字体
elif system_os == 'Linux':
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']  # Linux 字体
else:  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']  # Mac 字体

# 修复负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# Qwen Max API配置 (兼容OpenAI API)
Qwen_API_KEY = "sk-xxxxxxxxxxxxx"  # 替换为您的阿里云API密钥
Qwen_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Qwen_MODEL_NAME = "qwen-max-2025-01-25"  # Qwen Max模型名称

# 定义全局数据存储
global_data = {}

# ======================
# LangGraph状态机定义
# ======================
class AnalysisState(TypedDict):
    session_id: str
    user_query: str
    file_path: Optional[str]
    df: Optional[pd.DataFrame]
    table_metadata: Optional[str]
    generated_code: Optional[str]
    execution_result: Optional[str]
    visualization_type: Optional[str]
    suggested_questions: List[str]

# 创建Qwen Max客户端
def get_qwen_client():
    return ChatOpenAI(
        api_key=Qwen_API_KEY,
        base_url=Qwen_BASE_URL,
        model=Qwen_MODEL_NAME,
        temperature=0
    )

# ======================
# LangGraph工作流节点
# ======================
def save_uploaded_file(file: UploadFile):
    """保存上传文件并返回路径
    Args:
        file (UploadFile): FastAPI提供的上传文件对象，包含文件名和内容
        
    Returns:
        str: 保存后的文件完整路径
    """
    # 提取文件扩展名（如 .csv/.xlsx）
    file_ext = os.path.splitext(file.filename)[1]
    
    # 生成唯一UUID作为文件名前缀，避免文件名冲突
    file_id = str(uuid.uuid4())
    
    # 拼接完整保存路径：上传目录/UUID.扩展名
    file_path = f"{UPLOAD_DIR}/{file_id}{file_ext}"
    
    # 以二进制写入模式保存文件
    with open(file_path, "wb") as f:
        # 读取上传文件内容并写入目标路径
        f.write(file.file.read())
    
    # 返回保存后的文件路径供后续处理
    return file_path

def load_dataframe(state: AnalysisState):
    """加载数据文件到DataFrame
    Args:
        state (AnalysisState): 包含文件路径的状态字典
        
    Returns:
        AnalysisState: 更新后的状态字典，包含DataFrame和元数据
        
    Raises:
        HTTPException: 文件未上传/格式不支持/加载失败时抛出异常
    """
    print(f"进入load_dataframe，文件路径: {state['file_path']}") 
    # 检查文件路径是否存在
    if not state["file_path"]:
        raise HTTPException(status_code=400, detail="未上传文件")
    
    try:
        # 根据文件扩展名选择对应的读取方法
        if state["file_path"].endswith('.csv'):
            # 读取CSV文件
            # 尝试多种编码格式和解析选项
            encodings = ['utf-8', 'gb18030', 'gbk', 'big5', 'latin1']
            for encoding in encodings:
                try:
                    # 更新为Pandas新版本参数
                    df = pd.read_csv(
                        state["file_path"],
                        encoding=encoding,
                        on_bad_lines='skip',  # 替换error_bad_lines
                        engine='python'
                    )
                    if not df.empty:
                        break
                except Exception as e:
                    print(f"尝试编码 {encoding} 失败: {str(e)}")
                    continue
        elif state["file_path"].endswith(('.xlsx', '.xls')):
            # 读取Excel文件
            df = pd.read_excel(state["file_path"])
        else:
            # 不支持其他格式
            raise ValueError("不支持的文件格式")
            
        # 更新状态字典
        state["df"] = df  # 存储DataFrame
        # 生成表格元数据字符串,元数据包含列名和行数
        state["table_metadata"] = f"列名: {list(df.columns)}, 行数: {len(df)}"
        # 打印元数据信息
        print(state["table_metadata"])
        print(f"数据加载成功 - 列数: {len(df.columns)}, 行数: {len(df)}")
        return state
    except Exception as e:
        print(f"文件加载错误详情: {str(e)}")  # 添加详细错误打印
        # 捕获所有异常并转换为HTTP异常
        raise HTTPException(status_code=500, detail=f"文件加载失败: {str(e)}")

def parse_query(state: AnalysisState):
    """使用Qwen Max解析自然语言查询生成分析代码"""
    client = get_qwen_client()
    
    # 优化Prompt模板，明确要求JSON格式输出
    prompt = ChatPromptTemplate.from_template(
        """你是一个数据分析专家，请根据以下表格元数据和用户问题生成Python代码：
        表格元数据: {metadata}
        用户问题: {query}
        
        要求:
        1. 使用变量名df表示DataFrame（不要重新创建DataFrame）
        2. 结果赋值给result变量
        3. 使用matplotlib/seaborn生成可视化图表（如直方图、折线图、饼图等）
        4. 返回严格的JSON格式: {{"code": "完整代码", "visualization": "图表类型"}}
        5. 不要包含任何额外文本或注释，只返回JSON对象
        6. 确保代码安全可执行
        7. 不要包含plt.show()、plt.close()以及plt.savefig()语句
        """
    )
    
    chain = prompt | client | StrOutputParser()
    
    try:
        response = chain.invoke({
            "metadata": state["table_metadata"],
            "query": state["user_query"]
        })
        print(f"Qwen模型原始响应: {response}")
        
        # 优化响应解析逻辑 - 处理多种可能的响应格式
        response_dict = None
        
        # 情况1：响应是纯JSON（理想情况）
        if response.strip().startswith('{'):
            try:
                response_dict = json.loads(response)
            except json.JSONDecodeError:
                pass
        
        # 情况2：响应包含Python代码块（常见情况）
        if not response_dict and "```python" in response:
            # 提取完整的代码块内容
            code_block = re.search(r'```python(.*?)```', response, re.DOTALL)
            if code_block:
                code_content = code_block.group(1).strip()
                
                # 尝试从代码块中提取JSON结构
                json_match = re.search(r'\{.*\}', code_content, re.DOTALL)
                if json_match:
                    try:
                        response_dict = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        # 作为备选方案：直接使用整个代码块内容
                        state["generated_code"] = code_content
                        state["visualization_type"] = "auto"
                        return state
        
        # 情况3：响应包含JSON代码块
        if not response_dict and "```json" in response:
            json_block = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_block:
                try:
                    response_dict = json.loads(json_block.group(1).strip())
                except json.JSONDecodeError:
                    pass
        
        # 成功解析JSON的情况
        if response_dict:
            state["generated_code"] = response_dict.get("code", "")
            state["visualization_type"] = response_dict.get("visualization", "auto")
            return state
        
        # 所有解析尝试失败
        raise HTTPException(status_code=500, detail="无法解析模型响应")
        
    except Exception as e:
        print(f"查询解析错误详情: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询解析失败: {str(e)}")

# 在执行前验证代码缩进
def validate_code_indentation(code: str) -> str:
    """确保代码没有意外缩进"""
    lines = code.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # 跳过空行
        if not line.strip():
            cleaned_lines.append("")
            continue
            
        # 检查导入语句是否缩进
        if line.lstrip().startswith(("import ", "from ")) and line.startswith(" "):
            cleaned_lines.append(line.lstrip())
        else:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)


def execute_code(state: AnalysisState):
    """安全执行Python代码"""
    if not state["generated_code"]:
        return state
    
    try:
        # 创建安全环境
        local_vars = {"df": state["df"]}
        
        # 保留必要内置函数
        safe_builtins = {
            'print': print,
            'range': range,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            '__import__': __import__
        }
        
        global_vars = {
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "__builtins__": safe_builtins
        }
        
        # 添加必要的导入
        prepend_code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""
        
        # 清理生成的代码
        generated_code = state["generated_code"].strip()
        
        # 组合代码
        full_code = prepend_code + "\n" + generated_code
        
        # 打印最终代码（调试）
        print("="*50)
        print("最终执行代码:")
        print(full_code)
        print("="*50)
        
        # 执行代码
        exec(full_code, global_vars, local_vars)
        
        # 获取结果
        # 获取数值结果
        numerical_result = local_vars.get("result", None)
        #result = local_vars.get("result", "无结果")
        print(f"查询结果: {numerical_result}")

        # 处理可视化
        # 处理可视化 - 保存到内存缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 设置执行结果
        if numerical_result is not None:
            # 如果有数值结果，同时返回数值和图像
            state["execution_result"] = {
                "type": "composite",
                "data": {
                    "text": f"查询结果: {numerical_result}",
                    "image": img_base64
                }
            }
        else:
            # 只有图像结果
            state["execution_result"] = {
                "type": "image",
                "data": img_base64
            }
        
        return state
    except Exception as e:
        import traceback
        traceback.print_exc()
        state["execution_result"] = {
            "type": "error",
            "data": f"执行错误: {str(e)}"
        }
        return state

def generate_suggestions(state: AnalysisState):
    """使用Qwen Max生成后续问题建议
    Args:
        state (AnalysisState): 包含用户查询和执行结果的状态字典
        
    Returns:
        AnalysisState: 更新后的状态字典，包含建议的问题列表
        
    功能说明:
        1. 基于用户原始问题和执行结果，生成3个相关后续问题
        2. 处理模型返回的JSON或文本格式响应
        3. 返回最多3个建议问题
    """
    # 获取Qwen Max客户端实例
    client = get_qwen_client()
    
    # 构建Prompt模板，指导模型生成相关问题
    prompt = ChatPromptTemplate.from_template(
        """基于原始问题: {query}
        结合表格元数据: {metadata}
        请生成3个用户可能继续问的相关问题，并以JSON格式返回，包含一个名为'questions'的键，其值为问题列表。
        要求:
        1. 不要包含任何代码块标记(如```json)
        2. 直接返回JSON对象，不要额外说明
        示例:
        {{
            "questions": [
                "用户问题1",
                "用户问题2",
                "用户问题3"
            ]
        }}
        注意:
        1. 问题必须与表格内容相关
        2. 问题应简洁明了
        3. 问题不能与原始问题重复
        4. 问题必须是中文
        """
    )
    
    # 创建处理链：Prompt -> Qwen模型 -> 字符串输出解析器
    chain = prompt | client | StrOutputParser()
    
    # 简化结果摘要，限制在300字符以内
    metadata_summary = str(state["table_metadata"])[:300] + "..." if len(str(state["table_metadata"])) > 300 else str(state["table_metadata"])
    print(f"表格元数据摘要: {metadata_summary}")

    # 调用处理链，传入原始问题和结果摘要
    response = chain.invoke({
        "query": state["user_query"],
        "metadata": metadata_summary
    })
    print(f"模型原始响应: {response}")

    try:
        # 处理模型响应
        if response.startswith("{"):
            # JSON格式响应直接解析
            questions = json.loads(response)["questions"]
        else:
            # 文本格式响应尝试提取编号问题
            questions = [q.strip() for q in response.split("\n") if q.strip() and q.strip()[0].isdigit()]
            questions = [q.split(".", 1)[1].strip() for q in questions if "." in q]
        
        # 保存最多3个问题到状态
        state["suggested_questions"] = questions[:3]
        return state
    except:
        # 异常情况下返回空问题列表
        state["suggested_questions"] = []
        return state

# ======================
# LangGraph工作流构建
# ======================
workflow = StateGraph(AnalysisState)

# 添加节点
workflow.add_node("load_data", load_dataframe)
workflow.add_node("parse_query", parse_query)
workflow.add_node("execute_code", execute_code)
workflow.add_node("generate_suggestions", generate_suggestions)

# 设置工作流
workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "parse_query")
workflow.add_edge("parse_query", "execute_code")
workflow.add_edge("execute_code", "generate_suggestions")
workflow.add_edge("generate_suggestions", END)

# 编译工作流
analysis_workflow = workflow.compile()

# ======================
# FastAPI路由
# ======================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """文件上传接口
    Args:
        file (UploadFile): FastAPI提供的上传文件对象，包含文件名和内容
        
    Returns:
        dict: 包含会话ID和文件路径的字典
        
    Raises:
        HTTPException: 当上传过程出错时抛出500错误
    """
    try:
        # 调用save_uploaded_file函数保存文件到指定目录
        file_path = save_uploaded_file(file)
        
        # 生成唯一会话ID
        session_id = str(uuid.uuid4())
        
        # 在全局数据中记录文件路径
        global_data[session_id] = {"file_path": file_path}
        
        # 返回会话ID和文件路径给客户端
        return {"session_id": session_id, "file_path": file_path}
    except Exception as e:
        # 捕获所有异常并返回500错误
        raise HTTPException(500, str(e))

class AnalyzeRequest(BaseModel):
    session_id: str
    query: str


# 添加锁机制防止并发问题
import threading
global_data_lock = threading.Lock()

@app.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    session_id = request.session_id
    query = request.query
    
    # 检查会话ID是否存在于全局数据中
    #with global_data_lock:
    if session_id not in global_data:
        raise HTTPException(404, "会话不存在")
    
    # 获取会话数据
    session_data = global_data[session_id]
    file_path = session_data["file_path"]
    
    try:
        # 增强文件验证
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"文件不可读: {file_path}")
        
        # 正确初始化状态对象
        state = AnalysisState(
            session_id=session_id,
            user_query=query,
            file_path=file_path,
            df=None,
            table_metadata=None,
            generated_code=None,
            execution_result=None,
            visualization_type=None,
            suggested_questions=[]
        )
        
        print(f"即将开始工作流，文件路径: {state['file_path']}")
        
        # 执行工作流并捕获具体错误
        try:
            final_state = analysis_workflow.invoke(state)
        except Exception as workflow_error:
            # 记录详细错误信息
            error_trace = traceback.format_exc()
            print(f"工作流执行错误: {error_trace}")
            raise HTTPException(500, f"工作流执行失败: {str(workflow_error)}")
        
        # 更新全局数据（加锁保护）
        #with global_data_lock:
        session_data["last_result"] = final_state["execution_result"]
        session_data["suggestions"] = final_state["suggested_questions"]
        
        # 记录分析日志
        print(f"分析完成，会话ID: {session_id}，查询: {query}，结果: {final_state['execution_result']}, 建议问题: {final_state['suggested_questions']}")


        # 返回结构化响应
        return {
            "result": final_state["execution_result"],
            "suggested_questions": final_state["suggested_questions"],
            "generated_code": final_state["generated_code"]
        }
    
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    
    except PermissionError as e:
        raise HTTPException(403, str(e))
    
    except Exception as e:
        # 记录详细错误信息
        error_trace = traceback.format_exc()
        print(f"分析过程错误: {error_trace}")
        raise HTTPException(500, f"分析失败: {str(e)}")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 前端页面路由
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """前端页面路由
    Returns:
        FileResponse: 返回静态目录下的index.html文件
        
    功能说明:
        1. 作为应用的根路由("/")，返回前端页面
        2. 使用FileResponse返回静态文件
        3. 响应类型标记为HTMLResponse
    """
    return FileResponse("static/index.html")