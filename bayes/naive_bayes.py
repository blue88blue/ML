import re, collections

# 把语料库的单词全部抽取出来，转写成小写，并去掉单词中间的特殊符号
def words(file_dir):
    text = open(file_dir).read()
    return re.findall('[a-z]+',text.lower())

"""
    如果遇到一个语料库中没有的单词怎么办?
    假如说一个单词拼写正确，但是语料库中没有包含这个词，从而这个词也永远不会出现现在的训练集中。if 于是我们要返回出现这个词的概率是0
代表这个事件绝对不可能发生而在我们的概率模型中我们期望用一个很小的概率来代表这种情况。lambda:1

"""
def train(features):
    model = collections.defaultdict(lambda :1)
    for f in features:
        model[f] += 1
    return model

"""
    编辑距离:两个词之间的编辑距离定义为使用了几次插入(在词中插入一个单字母)，交换（交换相邻两个字母），
替换(把一个字母换成另一个)的操作从一个词变到另一个词
"""
# 返回所以与单词w编辑距离为1的集合

def editsl(word):
    n = len(word)
    return  set([word[0:i]+word[i+1:] for i in range(n)]+                           #deletion
                [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)]+       #transposition
                [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] +      #alteration
                [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet]        # insertion
                )

# 返回所有与单词w编辑距离为2的单词集合
# 在这些编辑距离小于2的中间，只把那些正确的词作为候选词

def know_edits2(word, WORDS):
    return set(e2 for e1 in editsl(word) for e2 in editsl(e1) if e2 in WORDS)


def known(words,WORDS):
    return set(w for w in words if w in WORDS)


def correct(words,WORDS):
    candidates = known([words],WORDS) or known(editsl(words),WORDS) or know_edits2(words,WORDS) or [words]
    return max(candidates,key=lambda w:WORDS[w])


if __name__ == "__main__":
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    WORDS = train(words('./big.txt'))

    word = input("input:")
    c_word = correct(word, WORDS)
    print(c_word)




