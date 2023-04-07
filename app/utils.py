from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pythainlp import word_tokenize, Tokenizer
from pythainlp.corpus import thai_stopwords
from rank_bm25 import BM25Okapi
import numpy as np
import mysql.connector
import requests # test
from mysql.connector import errorcode
import os

class Database:
    
    def __init__(self):
        self.host = "35.186.147.213"
        self.user="root"
        self.password="29062544"
        self.database="telecom_qa_db"
        self.mycursor = None 
        self.mydb = None
        
        
    def set_host(self, host):
        self.host = host
        
    def set_user(self, user):
        self.user = user
    
    def set_password(self, password):
        self.password = password
    
    def set_database(self, database):
        self.database = database

    def get_unanswer_question(self):
        mydb = mysql.connector.connect(host = self.host, user = self.user, passwd = self.password, database = self.database)

    
    def get_context_from_db(self):

        mydb = mysql.connector.connect(host = self.host, user = self.user, passwd = self.password, database = self.database)
        self.mydb = mydb

        mycursor = mydb.cursor()
        self.mycursor = mycursor

        mycursor.execute("SELECT * FROM telecom_qa_db.context_info")
        myresult = mycursor.fetchall()
        # print(myresult)

        # Grab only context
        cntx = [dt[1] for dt in myresult]
        
        # mycontext = []
        # cntx = []

        # # Grab context and created_at
        # for data in myresult:
        #     mycontext.append((data[1],data[2]))
            
        # # Grab only context
        # for dt in mycontext:
        #     cntx.append(dt[0].lower()) #lastest
            
        return cntx

    def insert_unanswerable_question_to_db(self, question):

        print("Logging : insert_unanswerable_question_to_db")

        try:
            sql = """INSERT INTO unanswerable_question (question) VALUES (%s)"""
            val = [(question)]
            self.mycursor.execute(sql, val)
            self.mydb.commit()
            print(self.mycursor.rowcount, "Record inserted successfully into table")

        except mysql.connector.Error as error:
            print("Failed to insert into MySQL table {}".format(error))
        
        # finally:
        #     if self.mydb.is_connected():
        #         self.mycursor.close()
        #         self.mydb.close()
        #         print("MySQL connection is closed")
    
class QAReaderModel:
    
    def __init__(self, model_path= os.getcwd()+"/app/model/finetune_11_03_2023_multilinguamodel-round3-hugedataset"):
        """_summary_

        Args:
            model_path (str): Directory containing the necessary tokenizer and model files.
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        # self.qa = None
        self.set_model(model_path)


    def qa(self):
        qa = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        return qa
    
    # def get_qa(self, model, tokenizer):
    #     qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
    #     return qa
        
    # def set_qa(self):
    #     self.qa = self.get_qa(self.model, self.tokenizer)
    
    def get_model(self, model_path):
        """
        Load a tokenizer and model using 'AutoTokenizer' and 'Automodel'

        Args:
            model_path (str): Directory containing the necessary tokenizer and model files.
        """

        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        print("modellllllll : ", model)

        # model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)   
        print("tokenizerrrrrrrrr : ", tokenizer)     
        
        return model, tokenizer
    
    def set_model(self, model_path):
        """
        Set a tokenizer and model using the 'self.get_model' method.

        Args:
            model_path (str): Directory containing the necessary tokenizer and model files.
        """
        self.model, self.tokenizer = self.get_model(model_path)
    
    def showData(self):
        # print(self.model, "\n", self.tokenizer, "\n", self.qa)
        # print(self.qa)
        qa = self.qa()
        print("qa info : {}".format(qa)) 
    
    
class QARetriever:
    
    def __init__(self, model_path=os.getcwd()+"/app/model/finetune_11_03_2023_multilinguamodel-round3-hugedataset"):
        """
        Define a QA Retriever model.
        Search for the associated context and return the best answer.

        Args:
            model_path (str): Directory containing the necessary tokenizer and model files.
        """
        
        self.answer = None
        self.question = None    # query question
        self.greeting = False
        self.thanks = False
        self.preprocessed_question = None # preprocessed query question
        self.contexts = None
        self.preprocessed_contexts = None
        self.qa_reader = QAReaderModel(model_path=model_path)
        self.tokenizer = self.qa_reader.tokenizer
        self.model = None
        self.qa = self.qa_reader.qa()
        self.database = Database() ##
        # self.best_answer = None # best score w/ best answer
        self.condition = False
        
    def set_question(self, question): # with api part
        self.question = question
        
    def set_preprocessed_question(self): 
        self.preprocessed_question = self.preprocess_query(self.question)

    def set_greeting(self): 
        greeting = ['สวัสดีครับ', 'สวัสดีค่ะ', 'สวัสดี', 'หวัดดีครับ', 'หวัดดีค่ะ', 'หวัดดี', 'Hi', 'hi', 'hello', 'Hello', 'ฮัลโหล','สวัสดีจ้า', 'ดีฮะ', 'ดีจ้า']
        just_greeting = False

        if str([self.question]) == str([val for val in greeting if self.question == val]):
            just_greeting = True
        else:
            pass

        self.greeting = just_greeting

    def set_thanks(self):  
        thanks = ['ขอบคุณค่ะ', 'ขอบคุณครับ', 'ขอบคุณ', 'แต๊ง', 'แต๊งกิ้ว', 'thank you', 'Thanks', 'Thank you', 'ขอบคุณฮะ', 'ขอบคุณงับ', 'Thx']
        just_thanks = False

        if str([self.question]) == str([val for val in thanks if self.question == val]):
            just_thanks = True

        else:
            pass

        self.thanks = just_thanks
        
    def set_preprocessed_contexts(self):
        self.preprocessed_contexts = self.preprocess_contexts()

        ##logging
        # print("preprocessed_contexts (in set_contexts) : {}".format(self.preprocessed_contexts))
         
    def preprocess_query(self, question):

        # start changing

        token_question = self.tokenizer(question).input_ids
        token_question = token_question[1:-1:] #remove special token

        decode_question = []
        for x in token_question:
            if x == 10: #Remove white space (token 10)
                pass
            else:
                decode = self.tokenizer.decode(x)
                decode_question.append(decode)
        already_question = decode_question
        # print("Token question: ", already_question)

        # # Remove Stopwords
        # stopwords = list(thai_stopwords())
        # already_question = [i for i in decode_question if i not in stopwords]
        # # print("Already Question: ", already_question)

        # end changing
        

        ## Tokenization
        # token_question = word_tokenize(question, engine="newmm",  keep_whitespace=False)
        
        # Remove stopwords
        # stopwords = list(thai_stopwords())
        
        return already_question
    
    def preprocess_contexts(self):

        all_context = self.database.get_context_from_db()

        # Set contexts obtained database
        self.contexts = all_context

        # start changing

        already_context = []
        for each_cntx in all_context:
            decode_context = []
            token_contexts = self.tokenizer(each_cntx.lower()).input_ids
            token_contexts = token_contexts[1:-1:] # remove special token
            for x in token_contexts:
                if x == 10: # white space removing (token 10)
                    pass
                else:
                    decode = self.tokenizer.decode(x)
                    decode_context.append(decode)
            already_context.append(decode_context)
            del decode_context

        # end changing
        
        # token_contexts = []
        
        # for context in all_context:
        #     token_contexts.append(word_tokenize(context, engine="newmm",  keep_whitespace=False))
            
        # ## Logging
        # # print("token_contexts : {}".format(token_contexts))
        
        # return token_contexts
        return already_context
    
    def retriever(self):
        
        # Initializing
        bm25 = BM25Okapi(self.preprocessed_contexts) 
        
        # Ranking of decuments
        doc_scores = bm25.get_scores(self.preprocessed_question)
        
        # Getting top 3 relevant context
        ranked_contexts = bm25.get_top_n(self.preprocessed_question, self.contexts, n=1)
        
         # Max to min bm25 score sorting
        top_n_score = np.flip(np.sort(doc_scores))
        
        # # Display ranking score
        # print("\nTop 3 most similar sentences in corpus:\n")
        # for score, ct in zip(top_n_score, ranked_contexts):
        #     print("(Score: {:.4f})".format(score),ct)
        # print("\n\n")
        
        return ranked_contexts

    def get_best_answers(self): 

        # start adding

        self.condition_checking()

        if self.condition == True: # found eng
            self.change_model()
        else:
            self.default_model()

        # stop adding

        self.set_preprocessed_contexts()
        self.set_preprocessed_question()
        
        ranked_contexts = self.retriever()

        answer = []

        # print("context in ranked_contexts")
        for context in ranked_contexts:
            
            ans = self.qa(question=self.question, context=context)
            answer.append((ans["score"], ans["answer"]))
            
        # Max to min answer score
        answer.sort(reverse=True)
        
        best_answer = max(answer) # score with answer
 


        # # testing start
            
        # if self.condition == True:
        #     API_URL = "https://api-inference.huggingface.co/models/wicharnkeisei/thai-xlm-roberta-base-squad2"
        #     headers = {"Authorization": f"Bearer {'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'}"}
            
        #     def query(payload):
        #         response = requests.post(API_URL, headers=headers, json=payload)
        #         return response.json()

        #     for context in ranked_contexts:
                
        #         output = query({
        #             "inputs": {
        #                 "question": "{}".format(self.question),
        #                 "context": "{}".format(context)
        #             },
        #         })
        #         answer.append((output['score'], output['answer']))

        #     answer.sort(reverse=True)
        #     best_answer = max(answer) # score with answer

        # else:
        #     for context in ranked_contexts:
        #         ans = self.qa(question=self.question, context=context)
        #         answer.append((ans["score"], ans["answer"]))
            
        # # Max to min answer score
        # answer.sort(reverse=True)
        
        # best_answer = max(answer) # score with answer

        # # testing end

        return best_answer
    
    
    def get_answer(self, question):
        
        self.set_question(question=question)   

        # start changing
        self.set_greeting()

        self.set_thanks()

        self.condition = False

        if self.greeting == True:
            answer = "สวัสดี ยินดีให้บริการ"
        elif self.thanks == True:
            answer = "ด้วยความยินดี"
        else:
            best_answer = self.get_best_answers()
            softmax_score = best_answer[0]

            print("softmaxxxxxxxx: ", softmax_score)
            
            threshold = 0.5

            if softmax_score < threshold:

                # print("Loggingggggggg_")

                # insert unanswerable question to db
                self.database.insert_unanswerable_question_to_db(question)

                # Here if else

                

                answer = "ขออภัย ไม่สามารถตอบคำถามนี้ได้ กรุณาถามคำถามใหม่ หรือติดต่อธุรการภาควิชาโทรคมนาคม"


        
            else:
                answer = best_answer[1] # best text answer


        # end changing

        
        return answer


    def default_model(self):
        self.tokenizer = self.qa_reader.tokenizer
        self.model = self.qa_reader.model
        self.qa = self.qa_reader.qa()

    def change_model(self, model_path="wicharnkeisei/thai-xlm-roberta-base-squad2"):
        self.set_changed_model(model_path)
        self.set_changed_qa()

    def get_changed_model(self, model_path):
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def set_changed_model(self, model_path):
        self.model, self.tokenizer = self.get_changed_model(model_path)

    def get_changed_qa(self):
        qa = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        return qa

    def set_changed_qa(self):
        self.qa = self.get_changed_qa()

    def condition_checking(self): 

        engs = ['ภาษาอังกฤษ', 'อิ้ง', 'อังกฤษ']

        tokenize_input = word_tokenize(self.question)

        condition = False

        for i in engs:
            for j in tokenize_input:
                if j == i:
                    condition = True
                    # print(i,j)
                    pass
        
        self.set_condition(condition)


    def set_condition(self, condition):
        self.condition = condition
            
        
    
        
        
        
        
        
        
        
        


