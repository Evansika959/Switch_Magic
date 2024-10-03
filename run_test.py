# pip install accelerate
from transformers.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-64")
model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-64", device_map="auto")

input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

