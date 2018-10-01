import Algorithmia

input = {
  "src":"Algorithmia is a marketplace for algorithms. The Technological Singularity will transform Society.",
  "format":"conll",
  "language":"english"
}
client = Algorithmia.client('simhEjDzj/ZTsQONGt9mcKTDnAt1')
result = client.algo('deeplearning/Parsey/1.0.4').pipe(input).result
print(result)

sentence_proper_nouns = []
for sentence_data in result['sentences']:
    for word in result['words']:
        if word['universal_pos'] == "PROPN":
            sentence_proper_nouns.append(word['form'])

print(sentence_proper_nouns)

# [Algorithmia, Society]