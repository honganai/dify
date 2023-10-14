import ast
import re

import requests
import json
import csv
import weaviate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from services.weaviate_search_service import WeaviateService
from flask import current_app
from core.index.vector_index.weaviate_vector_index import WeaviateConfig



from langchain.schema import SystemMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://2c6ee24b09816a6f14f95d1698b24ead.bitai.work/"
class_name="Dataset_keypoints_all_user"
prompt_template = """
You are a reading assistant that helps users make sense of content efficiently while stop wasting time  reading bloated information, aligning to the user's goals.
Goals:
{Goals To Insert}
Distill the content using the following steps and output according to the requirements defined step by step. Your output shall be using the same language as the source content.
1. First give out a concise summary in <Summary>.  
2. Determine the content type from the following list, and then move to step 4 and act accordingly: 
2.a. Factual information: the full content mainly presents facts and data
2.b. Opinion based: the full content mainly expresses reasoning and opinions on certain subjects.
2.c. Entertaining: produced for reading enjoyment
2.d. Research essay: show research work on certain topics that includes goals, methods, and results.
3. Distill valuable pieces of information using the following approach:
3.a. If the content type is Factual information, output the conclusions presented in the content in <Conclusions>. Then list out no more than 5 anecdotes, events, facts, or data pieces into <Data Sheet>, ruling out assertive or speculations. Go to Step 4.
3.b. If the content type is Opinion based, find out and the key argument logics and reasoning approaches that the author used and output to <Key Logics>. Then list conclusions into <Perspectives & Conclusions>. Skip <Data sheet>. Go to Step 4.
3.c. If the content type is Entertaining, explain why it is entertaining, and list out the memes and plots as <Plots>. Put interesting sentences as <Quotes>. Go to Step 4.
3.d. If the content type is Research Essay, list the key argument logics used into <Key logics>,  results and conclusions in <Perspectives & Conclusions>, and data or facts relevant to the key user goal in <Data sheet>. Go to Step 4.
4. Put impressive quotes for interviews if any in <Quotes>. If none there, skip this section.
5. Ask 4 further questions in <Unasked Questions> to expand the user's mindset ahead but still somewhat connected to the original content.
5.a List 1 questions or perspectives that were unsaid in the article, but the user should give more thoughts on when the user wants to have a 360 degree view on this topic or to take the discussion to the next level.
5.b List 1 counter arguments against the article's viewpoints for a balanced view.
5.c List 2 inspiring questions or perspectives that might look irrelevant to this specific topic on superficial level but somehow deeply connected by your knowledge of the world's hidden structures.

Refer to the following list of potential output sections. Each section starts with a new line.
<Summary>
<Content type>
<Plots>
<Key logics>
<Perspectives & Conclusions>.
<Data Sheet>
<Quotes>
<Further Questions>
"""

output_file=""

def request_openai(sys_prompt:str,assitant_prompt:str,user_prompt:str):
     #组装请求openai
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=4000, temperature=0, top_p=0.75, presence_penalty=0.5, frequency_penalty=0.5)

    response = chat([
         SystemMessage(content=sys_prompt),
         HumanMessage(content=user_prompt),
    ])
    return response.content

def save_data(data,output_file):
    #把数据保存进csv，csv中有两列
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


def read_file(filename: str,offset=0):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # next(reader, None)  # Skip the header
        content = [row[offset] for row in reader]  # Get the second column
    return content




def init_vector():
# 初始化向量数据库
    client = weaviate.Client(
        url = "http://8.217.23.8:8099",  # Replace with your endpoint
        auth_client_secret=weaviate.AuthApiKey(api_key="WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih"),  # Replace w/ your Weaviate instance API key
        additional_headers = {
            "X-OpenAI-Api-Key": "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",  # Replace with your inference API key
        }
    )

# def extract_keypoints(content, tag):
#     pattern = f'<{tag}>\n(.*?)\n\n'
#     result = re.search(pattern, content, re.S)
#     b = '\n'.join(list(map(lambda x: re.sub(r'^\d+\.\s', '', x), content.split('\n'))))
#
#     if result:
#         return result.group(1).split('\n')
#     else:
#         return []

def filter_prefix(text):
    return re.sub(r'^[-\w\d]+\.?\s', '', text)


def pattern_list(tag):
    return [
        f'<{tag}>\n?([^<]+)',
        f'<{tag}>\\n?([^<]+)',
    ]


def extract_keypoints(content, tag):
    re_result = None
    for pattern in pattern_list(tag):
        result = re.findall(pattern, content, re.S)
        if result:
            re_result = result[0]
    if not re_result:
        return []
    re_result = re_result.replace('\\n', '\n')
    re_result = re_result.strip()
    re_result = re_result.split('\n')
    re_result = list(map(filter_prefix, re_result))
    re_result = list(filter(lambda x: x, re_result))
    return re_result


def parse_content_to_keypoints(origin_content,llm_result):
    unasked_list = extract_keypoints(llm_result, 'Unasked Questions')
    further_questions = extract_keypoints(llm_result, 'Further Questions')
    question_list=[]
    keypoint_list=[]

    question_list.extend(unasked_list)
    question_list.extend(further_questions)
    keypoint_list.extend(extract_keypoints(llm_result, 'Conclusions'))
    keypoint_list.extend(extract_keypoints(llm_result, 'Quotes'))
    keypoint_list.extend(extract_keypoints(llm_result, 'Perspectives'))
    keypoint_list.extend(extract_keypoints(llm_result, 'Key logics'))
    keypoint_list.extend(extract_keypoints(llm_result, 'Perspectives & Conclusions'))
    keypoint_list.extend(extract_keypoints(llm_result, 'Data Sheet'))
    keypoint_list.extend(extract_keypoints(llm_result, 'Fact Sheet'))
    data=[origin_content,llm_result,keypoint_list,question_list]

    print(f"question_list:{question_list}")
    print(f"keypoint_list:{keypoint_list}")
    print(f"data:{data}")
    return data
    #写入到新的csv文件中 分别有两列



# 1. 读取txt文件
def vector_search(classname,keypoint,userid):
    config=WeaviateConfig(
        endpoint="http://8.217.23.8:8099",
        api_key="WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih",
        batch_size=10
    )
    weaviate=WeaviateService(config)

    return weaviate.search(class_name,keypoint,userid,3)

def vector_import(keypoint,userid):
    config=WeaviateConfig(
        endpoint="http://8.217.23.8:8099",
        api_key="WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih",
        open_api_key="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",
        batch_size=10

    )
    class_name="dataset_keypoints_all_user"
    weaviate=WeaviateService(config)
    # print(keypoint)
    # segment_uuid=""
    segment_uuid = weaviate.single_import_data(class_name,"", "",  keypoint, userid)
    return segment_uuid


def run_process_call_llm(filename):
    content_list=read_file(filename,2)
    for content in content_list:
        response=request_openai(sys_prompt=prompt_template,assitant_prompt="",user_prompt=content)
        #将question_list  keypoint_list保存到csv中
        data=[content,response]
        save_data(data,"文章distill.csv")

def run_process_parse_to_keypints(inputfilename,outputfilename):
    content_list = read_file(inputfilename,1)
    for content in content_list:
        data = parse_content_to_keypoints("",content)
        save_data(data,outputfilename)
        data=[]

def run_process_import_to_vector(inputfilename):
    keypoints_list = read_file(inputfilename,2)
    for keypoints in keypoints_list:
        keypoint_list = ast.literal_eval(keypoints)
        for keypoint in keypoint_list:
            print(keypoint)
            print(type(keypoint))
            #keypoints解析成数组
            uuid=vector_import(keypoint,"koman_test_1")
            print(uuid)

def run_process_search_vector(inputfilename):
    keypoints_list = read_file(inputfilename,2)
    for keypoints in keypoints_list:
        keypoint_list = ast.literal_eval(keypoints)
        for keypoint in keypoint_list:
            results = vector_search(class_name,keypoint,"koman_test_1")
            print(f"keypoint:{keypoint}")
            print(results)
            if len(results["data"]["Get"][class_name]) == 0:
                print("No results found")
            else:
                # print("Results:")
                for result in results["data"]["Get"][class_name]:
                    keypoint_result= result["keypoint"]
                    score= result["_additional"]["score"]
                    data = [keypoint,keypoint_result,score]
                    print(data)
                    save_data(data,"keypointswithsearch.csv")

def vetor_delete():
    pass

#执行函数
if __name__ == '__main__':
    #step1: 读取公众号csv 调用大模型生成keypoin 存入文件
    # run_process_call_llm("公众号文章.csv")
    # text = """
    # content='<Summary>\n本文讨论了SaaS企业在出海市场拓展中面临的困境以及如何破解这些困境。作者指出，选择适合自己的目标市场是关键，可以考虑成熟型市场和新兴市场。在立足海外市场方面，需要通过整合技术和服务提供更多价值给客户，并与大客户建立长期合作伙伴关系。制定战略和目标时要平衡长期战略和短期目标，并保持灵活性。最后，作者分享了一些出海经验，包括了解目标市场和客户、找到关键性人物、对产品和技术有信心等。\n\n<Content type>\nFactual information\n\n<Perspectives & Conclusions>\n- Choosing the right target market is crucial for SaaS companies to expand internationally.\n- Providing more value to customers through integrated technology and services is essential for establishing a foothold in overseas markets.\n- Building long-term partnerships with large clients can greatly benefit SaaS companies in terms of brand building and business expansion.\n- Balancing long-term strategies and short-term goals is important, and flexibility is necessary to adapt to changing market conditions.\n- Understanding the target market and customers, having confidence in products and technologies, and adapting growth strategies to different markets are key factors for success in international expansion.\n\n<Data Sheet>\n- The article mentions that the United States is the most mature SaaS market globally, while Southeast Asia, the Middle East, and Latin America are emerging markets with potential.\n- It highlights the importance of understanding market regulations and compliance requirements when entering different markets.\n- The article also emphasizes the significance of building partnerships with local companies and leveraging their networks for business growth.\n\n<Quotes>\n- "Choosing the right target market is crucial for SaaS companies to expand internationally."\n- "Providing more value to customers through integrated technology and services is essential for establishing a foothold in overseas markets."\n- "Building long-term partnerships with large clients can greatly benefit SaaS companies in terms of brand building and business expansion."\n\n<Further Questions>\n1. What are some other factors that SaaS companies should consider when choosing a target market for international expansion?\n2. How can SaaS companies effectively integrate technology and services to provide more value to customers in overseas markets?\n3. What are some potential challenges that SaaS companies may face when building long-term partnerships with large clients in international markets?\n4. How can SaaS companies adapt their growth strategies to different markets, such as mature markets versus emerging markets?' additional_kwargs={} example=False
    # """
    # step2: 读取文件解析大模型的keypoints 获取keypoints和question 保存进文件
    # run_process_parse_to_keypints("文章distill.csv","文章distillwithkeypoints.csv")
    # step3 :keypoints数据存入weaviate数据库
    # run_process_import_to_vector("文章distillwithkeypoints.csv")
    # step4: 读取weaviate数据库中的keypoints数据  通过weaviate搜索相似的keypoints
    run_process_search_vector("文章distillwithkeypoints.csv")

    #读取文件解析keypoints字段 存入向量数据库

    #parse_content_to_keypoints("" , text)


    # content = extract_content(text, 'content')
    # # print(content)
    # question_list = extract_keypoints(text, 'Further Questions')
    # conclusion_list = extract_keypoints(text, 'Conclusions')
    # quotes_list=extract_keypoints(text, 'Quotes')
    # plots_list=extract_keypoints(text, 'Plots')
    # key_logics_list = extract_keypoints(text, 'Key logics')
    # Perspectives_list = extract_keypoints(text, 'Perspectives & Conclusions')
    # data_sheet_list=extract_keypoints(text, 'Data Sheet')
    # print(Perspectives_list)
    # print(question_list)
    # print(quotes_list)

    # vector_search()
    # vector_import()
    # vetor_delete()




