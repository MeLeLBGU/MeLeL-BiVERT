import pickle
from transformers import MarianTokenizer, MarianMTModel, MBartForConditionalGeneration, MBart50TokenizerFast, GenerationConfig

class TransModel:

  def __init__(self, src, trg):
    self.model, self.tokenizer = self.__load_cache(src, trg)

  def __load_cache(self, src, trg,device):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model_file_name = f"path/opus-mt-{src}-{trg}.pkl"
    model = []
    tokenizer = []

    try:
      with (open(model_file_name, "rb")) as openfile:
        print("Found Model")
        while True:
          try:
            model.append(pickle.load(openfile))
          except Exception as exp:
            break
    except Exception as exp:
      print(exp)
      print(f"Model is downloading")
      model.append(MarianMTModel.from_pretrained(model_name, output_attentions = True).to(device))

      file = open(model_file_name, "wb")
      pickle.dump(model[0], file)
      file.close()

    print(f"Tokenizer is downloading")
    tokenizer.append(MarianTokenizer.from_pretrained(model_name))


    return model[0], tokenizer[0]
