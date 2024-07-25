import re
import spacy
import logging
import string
from typing import Optional
from graphviz import Digraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DependencyTreeNode:
    ''' Dependency tree node, representing a word in the sentence. '''

    def __init__(self, id=None, span=None, citation=None):
        self.id = id
        self.span = span
        self.children = []
        self.edge_values = {} # <child: edge_type>
        self.citation = citation # citation mark list
    
    def add_child(self, child, edge_value):
        self.children.append(child)
        self.edge_values[child] = edge_value
    
    def remove_child(self, child):
        if child in self.children:
            self.edge_values.pop(child)
            self.children.remove(child)
    
    def __repr__(self) -> str:
        if self.citation:
            return f'{self.span}:{self.id}:{self.citation}'
        else:
            return f'{self.span}:{self.id}'


def init_tree(child_dict: dict=None,
              span_dict: dict=None,
              cite_dict: dict=None,
              edge_dict: dict=None,
              root_id: int=0):
    ''' Construct a tree by dicts
    Args:
        - child_dict: id: [child_id]
        - span_dict: id: span
        - cite_dict: id: [citation]
        - edge_dict: child_id: edge_value
        - root_id
    Returns:
        - root node
    '''

    root = None

    def dfs(node: DependencyTreeNode, 
            child_dict: dict, 
            span_dict: dict, 
            cite_dict: dict,
            edge_dict: dict):
        ''' Construct nodes by dfs '''

        if node.id in child_dict:
            # this node has child or children

            for id in child_dict[node.id]:
                if id in cite_dict:
                    cur_node = DependencyTreeNode(id=id,
                                                  span=span_dict[id],
                                                  citation=cite_dict[id])
                else:
                    cur_node = DependencyTreeNode(id=id,
                                                  span=span_dict[id])

                dfs(cur_node, child_dict, span_dict, cite_dict, edge_dict)
                node.add_child(cur_node, edge_dict[id])
        
        return
    
    if child_dict is not None and span_dict is not None:
        if root_id in cite_dict:
            root = DependencyTreeNode(id=root_id,
                                      span=span_dict[root_id],
                                      citation=cite_dict[root_id])
        else:
            root = DependencyTreeNode(id=root_id,
                                      span=span_dict[root_id])
        dfs(root, child_dict, span_dict, cite_dict, edge_dict)
    
    return root


def print_tree(node: DependencyTreeNode, 
               name: str='tree', 
               dir: str='./trees',
               format: str='pdf'):
    ''' Draw the dependency tree of node '''

    G = Digraph()
        
    def add_edge_and_node(node: DependencyTreeNode):
        if node.citation:
            G.node(
                name=str(node.id), 
                label=f'{node.span}:{node.id}:{node.citation}', 
                color='red3',
                style='bold'
            )
        else:
            G.node(
                name=str(node.id), 
                label=f'{node.span}:{node.id}', 
                color='green3'
            )
        
        for child in node.children:
            G.edge(str(node.id), str(child.id), label=node.edge_values[child], color='black')
            add_edge_and_node(child)
    
    add_edge_and_node(node)

    G.render(name, directory=dir, format=format, view=False)


def generate_tree_text(node: DependencyTreeNode):
    ''' Convert the subtree of node into text '''

    output = {}
    output[node.id] = node.span

    for child in node.children:
        child_output = generate_tree_text(child)
        output = {**output, **child_output}
    
    return output


def find_node_by_id(node: DependencyTreeNode, id: int):
    ''' Search the target node according to id '''

    if node.id == id:
        return node
    
    for child in node.children:
        child_node = find_node_by_id(child, id)
        if child_node and child_node.id == id:
            return child_node
    
    return None


def find_target_child(node: DependencyTreeNode, id: int):
    ''' Find node's child whose subtree containing node with target id '''

    if node.id == id:
        return node
    
    for child in node.children:
        child_node = find_node_by_id(child, id)
        if child_node and child_node.id == id:
            return child
    
    return None


def find_node_father(node: DependencyTreeNode, root: DependencyTreeNode):
    ''' Find the ancestor of node '''

    if node == root:
        return root
    
    for child in root.children:
        if child == node:
            return root
        else:
            fnode = find_node_father(node, child)
            if fnode:
                return fnode
            
    return None


def find_lca(node: DependencyTreeNode,
             node1: DependencyTreeNode,
             node2: DependencyTreeNode):
    ''' Find the LCA node of node1 and node2 '''

    if node == node1 or node == node2:
        return node
    
    lca_list = [find_lca(child, node1, node2) for child in node.children]
    lca_list = [lca for lca in lca_list if lca is not None]
    if len(lca_list) > 1:
        # lca_list contains both node1 and node2
        return node
    
    # lca_list contains only node1 or node2, or contains no node
    return lca_list[0] if lca_list else None


def find_conj_sub_tree_between(node_i: DependencyTreeNode,
                               node_j: DependencyTreeNode,
                               father: DependencyTreeNode):
    ''' Find father node's child as a conjunction between node_i and node_j '''

    conj_types = ['cc', 'CC']
    
    conj_list = []
    for child in father.children:
        if (node_i.id < child.id < node_j.id or node_j.id < child.id < node_i.id) and \
            (child.span == ',' or father.edge_values[child] in conj_types):
            conj_list.append(child)

    return conj_list


def deep_copy(node: Optional[DependencyTreeNode]):
    ''' Deepcopy a tree '''

    if node is None:
        return None
    
    new_node = DependencyTreeNode(id=node.id,
                                  span=node.span,
                                  citation=node.citation)
    
    for child in node.children:
        new_child = deep_copy(child)
        new_node.add_child(new_child, node.edge_values[child])
    
    return new_node


def text_cleaning(sentence: str):
    ''' Remove punctuations in the sentence '''

    sentence = re.sub(r'[^\w\s\[\]\,]', '', sentence)

    return sentence


def get_words_and_citations(sentence: str):
    ''' Parsing a sentence into words and citation marks
    Returns:
        - word list without citation labels
        - words to which the citation labels are attached
    '''

    raw_seg_list = sentence.split()

    seg_list = []

    for raw_seg in raw_seg_list:
        citations = [(match.group(), match.start(), match.end()) for match in re.finditer(r"\[\d+\]", raw_seg)]

        if len(citations) == 0:
            # this segment has no citation label
            seg_list.append(raw_seg)
            continue

        if sum([len(citation[0]) for citation in citations]) == len(raw_seg):
            # the segment is all comprised of citation labels, like "[1][2]"
            for citation in citations:
                seg_list.append([int(citation[0][1: -1])])
        else:
            # the citation labels occur at the beginning of the sentence
            idx_list = [0]
            for citation in citations:
                idx_list = idx_list + [citation[1], citation[2]]
            idx_list = idx_list + [len(raw_seg)]

            for i in range(len(idx_list) - 1):
                seg = raw_seg[idx_list[i]: idx_list[i + 1]]
                if seg != '':
                    if re.search(r"\[\d+\]", seg):
                        seg_list.append([int(seg[1: -1])])
                    else:
                        seg_list.append(seg)
    
    # merge multi citations

    merge_seg_list = []
    cur_list = []

    for seg in seg_list:
        if isinstance(seg, list):
            cur_list.extend(seg)
        else:
            if cur_list:
                merge_seg_list.append(cur_list)
                cur_list = []
            merge_seg_list.append(seg)
    if cur_list:
        merge_seg_list.append(cur_list)
    
    # attach citations to words

    word_list = []
    cite_dict = {}
    cur_id = 0

    for idx, seg in enumerate(merge_seg_list):
        if isinstance(seg, list):
            if idx == 0:
                cite_dict[cur_id] = seg
            else:
                cite_dict[cur_id - 1] = seg
        else:
            word_list.append(seg.strip(string.punctuation))
            cur_id += 1
    
    return word_list, cite_dict


def alce_parse_citation(sentence):
    ''' ALCE parser (sentence-level) '''
    
    if re.search(r'\[\d+\]', sentence) is not None:
        raw_sentence = re.sub(r'\[\d+', '', re.sub(r' \[\d+', '', sentence)).replace(' |', '').replace(']', '')
        citations = [int(docId[1: -1]) for docId in re.findall(r'\[\d+\]', sentence)]

        return (raw_sentence, citations)
    else:
        # this sentence has no citation labels
        return (sentence, [])


def alice_parse_citation(sentence, index=0, draw=False):
    ''' ALiiCE parser (ours) '''

    if re.search(r'\[\d+\]', sentence) is None:
        # this sentence has no citation labels
        return (sentence, [])

    if len(sentence.split()) > 50:
        # prevent spacy dependency analysis from being too poor when the sentence is too long
        return alce_parse_citation(sentence)
    
    sentence = text_cleaning(sentence)
    word_list, raw_cite_dict = get_words_and_citations(sentence)

    dependency = spacy.load('en_core_web_sm')
    doc = dependency(' '.join(word_list))

    child_dict = {}
    span_dict = {}
    cite_dict = {} # token.i: [citation_id]
    edge_dict = {} # child_id: edge_value
    root_id = 0

    # check whether spacy's word segmentation is consistent with the word segmentation splited by space
    # according to 'raw_cite_dict' which provides the attatched word, we find a attached node in dependency tree, stored in cite_dict
    for word_idx, cite_group in raw_cite_dict.items():
        existed_list = []
        for token in doc:

            if word_list[word_idx] in token.text or token.text in word_list[word_idx]:
                existed_list.append(token.i)
        
        if existed_list:
            # if the distance is same, choose the previous one
            attached_idx = min(range(len(existed_list)), key=lambda i: abs(existed_list[i] - word_idx))
            cite_dict[existed_list[attached_idx]] = cite_group
        else:
            logging.warning(f'Error Occurred. Citation losted. :{sentence}')
            return (sentence, []) # degenerate to ALCE
    
    for token in doc:
        child_dict[token.i] = []
        span_dict[token.i] = token.text
        if token.dep_ == 'ROOT':
            root_id = token.i
    
    for token in doc:
        if token.dep_ != 'ROOT':
            child_dict[token.head.i].append(token.i)
            edge_dict[token.i] = token.dep_
    
    root = init_tree(child_dict,
                     span_dict,
                     cite_dict,
                     edge_dict,
                     root_id)
    
    if draw:
        print_tree(root, name=f'tree_{index}', dir='./trees')
    
    if root.citation:
        # if root has citation, there might be some errors in the citations and we change to use alce
        return alce_parse_citation(sentence)


    cite_texts = {} # text: [citation_id]
    for token_i, cite_group in cite_dict.items():

        root_i = deep_copy(root)
        node_i = find_node_by_id(root_i, token_i)

        if node_i is None:
            # this could happen if the func 'get_words_and_citations' has errors
            continue

        for token_j, _ in cite_dict.items():
            if token_j != token_i:
                node_j = find_node_by_id(root_i, token_j)
                if node_j is None:
                    continue

                node_lca = find_lca(root_i, node_i, node_j)

                if node_lca == node_i:
                    sub_tree_j = find_target_child(node_lca, node_j.id)
                    node_lca.remove_child(sub_tree_j)
                
                elif node_lca == node_j:
                    sub_tree_i = find_target_child(node_lca, node_i.id)
                    father_lca = find_node_father(node_lca, root_i)
                    father_lca.add_child(sub_tree_i, None)
                    father_lca.remove_child(node_lca)
                
                else:
                    sub_tree_i = find_target_child(node_lca, node_i.id)
                    sub_tree_j = find_target_child(node_lca, node_j.id)

                    conj_list = find_conj_sub_tree_between(sub_tree_i, sub_tree_j, node_lca)

                    if conj_list:

                        if sub_tree_i.id < sub_tree_j.id:
                            if node_lca == root_i and node_lca.edge_values[sub_tree_i] in ['prep', 'advcl'] and find_node_by_id(sub_tree_i, 0):
                                # this situation is sub_tree_i is a clause
                                # only save sub_tree_i
                                root_i = sub_tree_i
                            else:
                                # remove sub_tree_j and conj_list
                                node_lca.remove_child(sub_tree_j)
                                for conj_sub_tree in conj_list:
                                    node_lca.remove_child(conj_sub_tree)
                        else:
                            if node_lca == root_i:
                                if node_lca.edge_values[sub_tree_j] in ['prep', 'advcl'] and find_node_by_id(sub_tree_j, 0):
                                    # this situation is sub_tree_i is a clause
                                    # remove sub_tree_j and conj_list
                                    node_lca.remove_child(sub_tree_j)
                                    for conj_sub_tree in conj_list:
                                        node_lca.remove_child(conj_sub_tree)
                                else:
                                    # let sub_tree_i be the root_i
                                    root_i = sub_tree_i
                            else:
                                # only save sub_tree_i in node_lca's children
                                edge_value = node_lca.edge_values[sub_tree_i]
                                node_lca.children = []
                                node_lca.edge_values = {}
                                node_lca.add_child(sub_tree_i, edge_value)

                    else:

                        if node_lca == root_i:
                            # only save sub_tree_i
                            root_i = sub_tree_i
                        else:
                            # just mask sub_tree_j
                            node_lca.remove_child(sub_tree_j)
        
        if draw:
            print_tree(root_i, name=f'tree_{index}_{token_i}', dir='./trees')

        output = generate_tree_text(root_i)
        sorted_output = dict(sorted(output.items()))
        cite_text = [token_span for _, token_span in sorted_output.items()]
        cite_text = ' '.join(cite_text)

        cite_texts[cite_text] = cite_group

    return cite_texts



if __name__ == '__main__':
    # examples collected from wikipedia
    wiki_sentences = [
        'It is consistently ranked as one of the ten most popular websites in the world, and as of 2024 is ranked the fifth most visited website on the Internet by Semrush[1], and second by Ahrefs[2].',
        'Wales is credited with defining the goal of making a publicly editable encyclopedia[3][4], while Sanger is credited with the strategy of using a wiki to reach that goal[5].',
        'The domains wikipedia.org and wikipedia.com (later redirecting to wikipedia.org) were registered on January 13, 2001[6], and January 12, 2001[7], respectively.',
        'Wikipedia was launched on January 15, 2001[8] as a single English-language edition at www.wikipedia.com[9], and was announced by Sanger on the Nupedia mailing list[10].',
        'As of January 2023, 55,791 English Wikipedia articles have been cited 92,300 times in scholarly journals[11], from which cloud computing was the most cited page[12].',
        'It is widely seen as a resource-consuming scenario where no useful knowledge is added[13], and criticized as creating a competitive[14] and conflict-based editing culture associated with traditional masculine gender roles[15][16].',
        'Wikipedia\'s community has been described as cultlike[17], although not always with entirely negative connotations[18].',
        'For instance, Meta-Wiki provides important statistics on all language editions of Wikipedia[19], and it maintains a list of articles every Wikipedia should have[20].',
        'The findings by Nature were disputed by Encyclopædia Britannica[21][22], and in response, Nature gave a rebuttal of the points raised by Britannica[23].',
        'The presence of politically, religiously, and pornographically sensitive materials in Wikipedia has led to the censorship of Wikipedia by national authorities in China[24] and Pakistan[25], amongst other countries[26][27][28].',
        'In 2019, the level of contributions were reported by the Wikimedia Foundation as being at $120 million annually[29], updating the Jaffe estimates for the higher level of support to between $3.08 million and $19.2 million annually[30].',
        'The Polish-language version from 2006 contains nearly 240,000 articles[31], the German-language version from 2007/2008 contains over 620,000 articles[32], and the Spanish-language version from 2011 contains 886,000 articles[33].',
        'Russians have developed clones called Runiversalis[34] and Ruwiki[35].',
        'In addition to logistic growth in the number of its articles[36], Wikipedia has steadily gained status as a general reference website since its inception in 2001[37].',
        'Wikipedia has also been used as a source in journalism[38][39], often without attribution, and several reporters have been dismissed for plagiarizing from Wikipedia[40][41][42][43].',
        'In 2015, Wikipedia was awarded both the annual Erasmus Prize, which recognizes exceptional contributions to culture, society or social sciences[44], and the Spanish Princess of Asturias Award on International Cooperation[45].',
        'One of the most important areas is the automatic detection of vandalism[46][47] and data quality assessment in Wikipedia[48][49].',
        'Several free-content, collaborative encyclopedias were created around the same period as Wikipedia (e.g. Everything2)[50], with many later being merged into the project (e.g. GNE)[51].'
    ]

    for index, sentence in enumerate(wiki_sentences):
        alice_parse_citation(sentence, index=index, draw=True)


