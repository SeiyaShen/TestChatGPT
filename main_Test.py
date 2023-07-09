# main.py
# 最快捷测试embidding的主函数
from chatgpt_utils import get_answer

def main():
    #text = "Can I get information on cable company tax revenue?"
    text = "how old are you ?"
    engine = "Seiya-embedding-ada"
     
    #rtn = search_docs(text,top_n=1)
    #print("Embedding for the given text: " )
    #print(rtn) 

    rtn = get_answer("When is the date today?")
    print(rtn)

if __name__ == "__main__":
    main()
