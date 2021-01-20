from typing import List
import os
import torch

class Solution:
    # --------------------
    # 在此填写 学号 和 用户名
    # --------------------
    ID = "18307130370"
    NAME = "贺劲洁"

    # --------------------
    # 对于下方的预测接口，需要实现对你模型的调用：
    #
    # 要求：
    #    输入：一组句子
    #    输出：一组预测标签
    #
    # 例如：
    #    输入： ["我爱北京天安门", "今天天气怎么样"]
    #    输出： ["SSBEBIE", "BEBEBIE"]
    # --------------------

    # --------------------
    # 一个样例模型的预测
    # --------------------
    def example_predict(self, sentences: List[str]) -> List[str]:
        from .example_model import ExampleModel

        model = ExampleModel()
        results = []
        for sent in sentences:
            results.append(model.predict(sent))
        return results

    # --------------------
    # HMM 模型的预测接口
    # --------------------
    def hmm_predict(self, sentences: List[str]) -> List[str]:
        from wordseg.HMM2 import HmmModel
        hmm = HmmModel()
        hmm.load_txt(save_dir=".\\wordseg\\\\ModelSave\\hmm_model", prefix="12_total_")
        print("emit:",len(hmm.emit))

        result_list = []
        for sent in sentences:
            result = hmm.predict(sent)
            # print(result)
            result_list.append(result)
        
        # print("HMM result:",result_list)
        return result_list

    # --------------------
    # CRF 模型的预测接口
    # --------------------
    def crf_predict(self, sentences: List[str]) -> List[str]:
        from wordseg.CRF import CRF_model

        crf = CRF_model()
        # load
        LOAD_DIR = ".\\wordseg\\ModelSave\\crf_model"

        # load_prefix =  "train12-train12_eval-win5_no7[94.6670994817812]"
        load_prefix = "train12-train12_eval-win5_no7[94.78766940654091]"
        crf.load_model(LOAD_DIR,load_prefix)
        print("crf load path:", load_prefix)

        result_list = []
        for sen in sentences:
            result = ""
            predict = crf.predict(sen)
            for p in predict:
                result += p
            result_list.append(result)
        return result_list
    

    # --------------------
    # DNN 模型的预测接口
    # --------------------
    def dnn_predict(self, sentences: List[str]) -> List[str]:
        # from .BiLSTM import BiLSTM_CRF2
        from wordseg.BiLSTM_CRF.BiLSTM import BiLSTM_CRF2

        print("dir:", os.getcwd())
        # lstm = BiLSTM_CRF2()
         # load
        LOAD_DIR = ".\\wordseg\\ModelSave\\BiLSTM"

        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        PAD_TAG = "<PAD>"

        tag_to_ix = {"B": 0, "I": 1, "E": 2, "S":3,  START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}
        ix_to_tag = dict([(v,k) for (k,v) in tag_to_ix.items()])
        # print("word_to_ix",word_to_ix)

        model_path = os.path.join(LOAD_DIR, "train12-train12_eval-15-32[0.92]")
        word_ix_path = os.path.join(LOAD_DIR, "train12-train12_eval-30-8test_word_to_ix.txt")

        f = open(word_ix_path,"r")
        word_to_ix = eval(f.read())
        print("word_to_ix size:",len(word_to_ix) )
        f.close()
        # model = torch.load(model_path)
        model = BiLSTM_CRF2(len(word_to_ix), tag_to_ix, 15, 32)
        checkpoint = torch.load(os.path.join(LOAD_DIR, "state_dict.lstm"))
        model.load_state_dict(checkpoint['state_dict'])
        

        result_list = []
        with torch.no_grad():
            for sen in sentences:
                precheck_sent = prepare_sequence(sen, word_to_ix)
                _, predict_ix = model(precheck_sent)
                result = ""
                for ix in predict_ix:
                    result += ix_to_tag[ix]
                result_list.append(result)

        return result_list


def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    # return torch.tensor(idxs, dtype=torch.long)
    for w in seq:
        if w not in to_ix:
            # print("before:", len(to_ix))
            to_ix[w] = len(to_ix)
            # print("add ",w)
            # print("after:", len(to_ix))

    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
