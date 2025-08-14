<img width="3098" height="2185" alt="dfd6d48029e56d025751aa198fa57c7b" src="https://github.com/user-attachments/assets/1861041f-98b5-4203-8206-46e3ddca8863" />

<img width="2201" height="2034" alt="70f2b6657c2650edc3e3d09e410e1321" src="https://github.com/user-attachments/assets/1e95eb71-4e0e-4835-875c-6c1f38f27aa4" />

<img width="2364" height="1901" alt="325602bf2ca9b5583cdbc153117f89be" src="https://github.com/user-attachments/assets/4aa6684f-090b-4cad-8b04-bfa57c87bdf9" />

<img width="2384" height="2006" alt="a97957a57eed904f35f7481d5dba91f0" src="https://github.com/user-attachments/assets/b2c7fbd2-d05d-4c93-b8e5-c25f55fc366e" />

<img width="1969" height="2034" alt="6d8395927dfcca4ae02457f541c7ab47" src="https://github.com/user-attachments/assets/ae55175a-d538-49b1-9b30-ed9ee913a05a" />

核心功能模块 

一、应用初始化与配置 
• 使用FastAPI框架构建Web服务 
• 配置CORS跨域支持 
• 设置文件上传目录 
• 配置matplotlib中文字体支持（适配Windows/Linux/macOS） 
• 配置Qwen Max API密钥和模型参数  

二、数据结构定义 
•  AnalysisState : 使用TypedDict定义LangGraph状态机的数据结构，包含会话ID、用户查询、文件路径、DataFrame、元数据、生成代码、执行结果等字段  

三、核心处理流程（LangGraph工作流） 
LangGraph工作流包含四个核心节点： 
        1.  load_dataframe: 
加载上传的CSV/Excel文件到pandas DataFrame ◦ 支持多种编码格式（utf-8, gb18030, gbk等） ◦ 自动生成表格元数据（列名、行数）   
        2.  parse_query: 
使用Qwen Max解析自然语言查询 ◦ 构造Prompt模板，包含表格元数据和用户问题 ◦ 要求模型生成Python代码和可视化类型 ◦ 处理多种响应格式（纯JSON、代码块、JSON块）   
        3.  execute_code: 
安全执行生成的Python代码 ◦ 创建受限执行环境 ◦ 处理可视化输出（保存为图片文件） ◦ 处理表格和文本结果   
        4.  generate_suggestions: 
生成后续问题建议 ◦ 基于用户查询和执行结果，生成3个相关问题    

四、API接口 
1.  文件上传接口 ( /upload ) 
接收上传的CSV/Excel文件 ◦ 生成唯一会话ID ◦ 保存文件路径到全局数据   
2.  数据分析接口 ( /analyze )  
接收会话ID和自然语言查询 ◦ 执行完整的LangGraph工作流 ◦ 返回分析结果、建议问题和生成代码   
3.  前端页面路由 ( / ) 
返回静态HTML页面

五、特色功能 
• 多格式支持: 支持CSV和Excel文件上传 
• 智能编码处理: 自动尝试多种编码格式读取CSV文件 
• 可视化生成: 自动生成matplotlib/seaborn图表 
• 后续问题推荐: 基于分析结果智能推荐相关问题 
• 安全执行环境: 在受限环境中执行生成的代码  

六、技术架构 
• 后端框架: FastAPI 
• 大模型服务: Qwen Max (阿里云) 
• 数据处理: pandas 
• 可视化: matplotlib/seaborn 
• 工作流管理: LangGraph 
• 前端: 静态HTML页面  整体而言，这是一个功能完整、结构清晰的数据分析服务，充分利用了大模型的自然语言理解能力，为用户提供便捷的数据分析体验。
