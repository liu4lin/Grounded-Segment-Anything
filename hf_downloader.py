from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

exit()

from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir='/data-cbs1/ubuntu/checkpoints/hf')

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir='/data-cbs1/ubuntu/checkpoints/hf')
        