#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "tokenizer.h"
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/tokenizer.hpp>
namespace baidu {
    int WordpieceTokenizer::init(SimpleDict *p_simple_dict) {
        if (NULL == p_simple_dict) {
            return -1;
        }
        _p_dict = p_simple_dict;
        _unk_token = "[UNK]";
        _max_input_chars_per_word = 200;
        return 0;
    }
    /*
       """Tokenizes a piece of text into its word pieces.
       This uses a greedy longest-match-first algorithm to perform tokenization
       using the given vocabulary.
       For example:
       input = "unaffable"
       output = ["un", "##aff", "##able"]
       Args:
       text: A single token or whitespace separated tokens. This should have
       already been passed through `BasicTokenizer.
       Returns:
       A list of wordpiece tokens.
       """
     */
    int WordpieceTokenizer::tokenize(const std::string &input_string, std::vector<std::string> &split_tokens) {
        if (input_string.empty()) {
            return 0;
        }
        unsigned char p = (unsigned char) input_string[0];
        if (p >= 0x80) {
            //chinese and other char
            std::cout << "chinese and other:" << input_string << std::endl;
            if (_p_dict->search_dict(input_string) >= 0) {
                split_tokens.push_back(input_string);
            }
            else {
                split_tokens.push_back(_unk_token);
            }
            return 0;
        }
        std::vector<std::string> clean_tokens;
        std::string delm = " \n\r\t";
        boost::split(clean_tokens, input_string, boost::is_any_of(delm), boost::token_compress_on);
        for (std::vector<std::string>::iterator iter = clean_tokens.begin(); iter != clean_tokens.end(); ++iter) {
            int token_len = iter->length();
            if (token_len > _max_input_chars_per_word) {
                split_tokens.push_back(_unk_token);
                continue;
            }
            bool is_bad = false;
            int start = 0;
            std::vector<std::string> sub_tokens;
            while (start < token_len) {
                int end = token_len;
                std::string cur_substr = "";
                int sub_len = end - start;
                while (start < end && sub_len > 0) {
                    std::string substr = iter->substr(start, sub_len);
                    if (start > 0) {
                        substr = "##" + substr;
                    }
                    if (_p_dict->search_dict(substr) >= 0) {
                        cur_substr = substr;
                        break;
                    }
                    sub_len -= 1;
                }
                if (cur_substr.empty()) {
                    is_bad = true;
                    break;
                }
                sub_tokens.push_back(cur_substr);
                start += sub_len;
            }
            if (is_bad) {
                split_tokens.push_back(_unk_token);
            }
            else {
                split_tokens.insert(split_tokens.end(), sub_tokens.begin(), sub_tokens.end());
            }
        }
        return 0;
    }
    int BasicTokenizer::init(bool do_lower_case) {
        _do_lower_case = do_lower_case;
        //gen punctuation
        std::stringstream ss;
        for (int i = 33; i <= 47; ++i){
            ss << char(i);
        }
        for (int i = 58; i <= 64; ++i){
            ss << char(i);
        }
        for (int i = 91; i <= 96; ++i){
            ss << char(i);
        }
        for (int i = 123; i <= 126; ++i){
            ss << char(i);
        }
        ss >> _punctuation;
        std::cout << "_punctuation:" << _punctuation << std::endl;
        return 0;
    }
    /*
    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
     */
    int BasicTokenizer::tokenize(const std::string &input_string, std::vector<std::string> &split_tokens) {
        split_tokens.clear();
        std::vector<std::string> orig_tokens;
        if (utf8_to_char(input_string, orig_tokens) != 0) {
            return -1;
        }
        for (std::vector<std::string>::iterator iter = orig_tokens.begin(); iter != orig_tokens.end(); ++iter) {
            //filter 
            if (iter->empty()) {
                continue;
            }
            if (*iter == " " || *iter == "\r" || *iter == "\n" || *iter == "\t" || *iter == "\r\n") {
                continue;
            }
            unsigned char p = (unsigned char) (*iter)[0];
            if (p < 0x80) {
                //not chinese
                //to lower
                if (_do_lower_case) {
                    std::transform(iter->begin(), iter->end(), iter->begin(), ::tolower);
                    //_run_strip_accents?
                }
                //_run_split_on_punc
                //std::vector<std::string> sub_tokens;
                typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
                boost::char_separator<char> sep("", _punctuation.c_str(), boost::keep_empty_tokens);
                tokenizer tokens(*iter, sep);
                for (tokenizer::iterator tok_iter = tokens.begin();
                        tok_iter != tokens.end(); ++tok_iter) {
                    split_tokens.push_back(*tok_iter);
                    //std::cout << "<" << *tok_iter << "> ";
                }
                //boost::split(sub_tokens, *iter, boost::is_any_of(_punctuation), boost::token_compress_on);
                //split_tokens.insert(split_tokens.end(), sub_tokens.begin(), sub_tokens.end());  //append tokens
                continue;
            }
            //chinese
            split_tokens.push_back(*iter);
        }
        return 0;
    }
    int FullTokenizer::init(std::string &dict_file, bool do_lower_case) {
        int ret = _simple_dict.load_dict(dict_file);
        if (ret != 0) {
            return ret;
        }
        _do_lower_case = do_lower_case;
        ret = _wordpiece_tokenizer.init(&_simple_dict);
        if (ret != 0) {
            return ret;
        }
        ret = _basic_tokenizer.init(do_lower_case);
        if (ret != 0) {
            return ret;
        }
        return 0;
    }
    int FullTokenizer::tokenize(const std::string &input_string, std::vector<std::string> &split_tokens) {
        split_tokens.clear();
        std::vector<std::string> basic_tokens;
        std::vector<std::string> wordpiece_tokens;
        _basic_tokenizer.tokenize(input_string, basic_tokens);
        std::cout << "basic res:" << std::endl;
        for (auto& i: basic_tokens) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        for (std::vector<std::string>::const_iterator iter = basic_tokens.begin(); iter != basic_tokens.end(); ++iter) {
            wordpiece_tokens.clear();
            _wordpiece_tokenizer.tokenize(*iter, wordpiece_tokens);
            //append tokens
            split_tokens.insert(split_tokens.end(), wordpiece_tokens.begin(), wordpiece_tokens.end());
        }
        return 0;
    }
    int FullTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens, std::vector<int> &ids) {
        ids.clear();
        for (std::vector<std::string>::const_iterator iter = tokens.begin(); iter != tokens.end(); ++iter) {
            ids.push_back(_simple_dict.search_dict(*iter));
        }
        return 0;
    }
    int FullTokenizer::convert_ids_to_tokens(const std::vector<int>& ids, std::vector<std::string> &tokens) {
        tokens.clear();
        int len = ids.size();
        for (int i = 0; i < len; ++i) {
            tokens.push_back(_simple_dict.search_invert_dict(ids[i]));
        }
        return 0;
    }
}

int main()
{
    baidu::FullTokenizer a;
    std::string vocab_file = "vocab.txt";
    a.init(vocab_file, false);
    std::string s1 = "肺炎;石家庄;美元;法制;习近平;特朗普;时事;社会万象";
    std::string s2 = "公共生活";
    std::vector<std::string> s1_out;
    a.tokenize(s1, s1_out);
    std::vector<std::string> s2_out;
    a.tokenize(s2, s2_out);
    std::vector<std::string> all_tokens;
    all_tokens.emplace_back("[CLS]");
    for (auto& i: s1_out) {
        all_tokens.emplace_back(i);
    }
    all_tokens.emplace_back("[SEP]");
    for (auto& i: s2_out) {
        all_tokens.emplace_back(i);
    }
    all_tokens.emplace_back("[SEP]");
    
    std::vector<int> ids;
    a.convert_tokens_to_ids(all_tokens, ids);
    for (auto& i: ids) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}
