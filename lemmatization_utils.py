import re

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet as wn


class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """

    CONTRACTIONS2 = [
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(d)(?#X)('ye)\b",
        r"(?i)\b(gim)(?#X)(me)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(lem)(?#X)(me)\b",
        r"(?i)\b(mor)(?#X)('n)\b",
        r"(?i)\b(wan)(?#X)(na)\s",
    ]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b", r"(?i)\b(wha)(t)(cha)\b"]


class TreebankWordTokenizer:
    """
    The Treebank tokenizer uses regular expressions to tokenize text as in
    Penn Treebank. This is the method. It assumes that thetext has already been
    segmented into sentences.
    This tokenizer performs the following steps:
    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line
    """

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(u'([«“‘„]|[`]+)', re.U), r' \1 '),
        (re.compile(r'^\"'), r'``'),
        (re.compile(r'(``)'), r' \1 '),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r'\1 `` '),
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r'([^\.])(\.)([\]\)}>"\'' u'»”’ ' r']*)\s*$', re.U), r'\1 \2 \3 '),
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r'\1 \2\3 ',
        ),  # Handles the final period.
        (re.compile(r'[?!]'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> ')

    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile(r'\('), '-LRB-'),
        (re.compile(r'\)'), '-RRB-'),
        (re.compile(r'\['), '-LSB-'),
        (re.compile(r'\]'), '-RSB-'),
        (re.compile(r'\{'), '-LCB-'),
        (re.compile(r'\}'), '-RCB-'),
    ]

    DOUBLE_DASHES = (re.compile(r'--'), r' -- ')

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(u'([»”’]+)', re.U), r' \1 '),
        (re.compile(r'"'), " '' "),
        (re.compile(r'(\S)(\'\')'), r'\1 \2 '),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text, convert_parentheses=False, return_str=False):
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        # Optionally convert parentheses
        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r' \1 \2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r' \1 \2 ', text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self._contractions.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text if return_str else text.split()


_treebank_word_tokenizer = TreebankWordTokenizer()

def word_tokenize(text, language='english', preserve_line=False):
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [token for sent in sentences
            for token in _treebank_word_tokenizer.tokenize(sent)]


SS_PARAMETERS_TYPE_MAP = {'definition':str,
                          'lemma_names':list, # list(str)
                          'examples':list,
                          'hypernyms':list,
                          'hyponyms': list,
                          'member_holonyms':list,
                          'part_holonyms':list,
                          'substance_holonyms':list,
                          'member_meronyms':list,
                          'substance_meronyms': list,
                          'part_meronyms':list,
                          'similar_tos':list
                          }


def remove_tags(text: str) -> str:
    """ Removes <tags> in angled brackets from text. """

    tags = {i:" " for i in re.findall("(<[^>\n]*>)",text.strip())}
    no_tag_text = reduce(lambda x, kv:x.replace(*kv), tags.iteritems(), text)
    return " ".join(no_tag_text.split())


def offset_to_synset(offset):
    """
    Look up a synset given offset-pos
    (Thanks for @FBond, see http://moin.delph-in.net/SemCor)
    >>> synset = offset_to_synset('02614387-v')
    >>> print '%08d-%s' % (synset.offset, synset.pos)
    >>> print synset, synset.definition
    02614387-v
    Synset('live.v.02') lead a certain kind of life; live in a certain style
    """

    return wn._synset_from_pos_and_offset(str(offset[-1:]), int(offset[:8]))


def semcor_to_synset(sensekey):
    """
    Look up a synset given the information from SemCor sensekey format.
    (Thanks for @FBond, see http://moin.delph-in.net/SemCor)
    >>> ss = semcor_to_offset('live%2:42:06::')
    >>> print '%08d-%s' % (ss.offset, ss.pos)
    >>> print ss, ss.definition
    02614387-v
    Synset('live.v.02') lead a certain kind of life; live in a certain style
    """

    return wn.lemma_from_key(sensekey).synset


def semcor_to_offset(sensekey):
    """
    Converts SemCor sensekey IDs to synset offset.
    >>> print semcor_to_offset('live%2:42:06::')
    02614387-v
    """

    synset = wn.lemma_from_key(sensekey).synset
    offset = '%08d-%s' % (synset.offset, synset.pos)
    return offset


porter = PorterStemmer()
wnl = WordNetLemmatizer()


def lemmatize(ambiguous_word: str, pos: str = None, neverstem=False,
              lemmatizer=wnl, stemmer=porter) -> str:
    """
    Tries to convert a surface word into lemma, and if lemmatize word is not in
    wordnet then try and convert surface word into its stem.
    This is to handle the case where users input a surface word as an ambiguous
    word and the surface word is a not a lemma.
    """

    # Try to be a little smarter and use most frequent POS.
    pos = pos if pos else penn2morphy(pos_tag([ambiguous_word])[0][1],
                                     default_to_noun=True)
    lemma = lemmatizer.lemmatize(ambiguous_word, pos=pos)
    stem = stemmer.stem(ambiguous_word)
    # Ensure that ambiguous word is a lemma.
    if not wn.synsets(lemma):
        if neverstem:
            return ambiguous_word
        if not wn.synsets(stem):
            return ambiguous_word
        else:
            return stem
    else:
        return lemma


def penn2morphy(penntag, returnNone=False, default_to_noun=False) -> str:
    """
    Converts tags from Penn format (input: single string) to Morphy.
    """
    morphy_tag = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        if returnNone:
            return None
        elif default_to_noun:
            return 'n'
        else:
            return ''


def lemmatize_sentence(sentence: str, neverstem=False, keepWordPOS=False,
                       tokenizer=word_tokenize, postagger=pos_tag,
                       lemmatizer=wnl, stemmer=porter) -> list:

    words, lemmas, poss = [], [], []
    for word, pos in postagger(tokenizer(sentence)):
        pos = penn2morphy(pos)
        lemmas.append(lemmatize(word.lower(), pos, neverstem,
                                lemmatizer, stemmer))
        poss.append(pos)
        words.append(word)

    if keepWordPOS:
        return words, lemmas, [None if i == '' else i for i in poss]

    return lemmas

def synset_properties(synset: "wn.Synset", parameter: str):
    """
    Making from NLTK's WordNet Synset's properties to function.
    Note: This is for compatibility with NLTK 2.x
    """

    return_type = SS_PARAMETERS_TYPE_MAP[parameter]
    func = 'synset.' + parameter

    return eval(func) if isinstance(eval(func), return_type) else eval(func)()

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))
