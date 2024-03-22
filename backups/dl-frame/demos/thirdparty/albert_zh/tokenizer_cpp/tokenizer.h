#ifndef DEMO_TOKENIZATION_H
#define DEMO_TOKENIZATION_H
#include <string>
#include <iostream>
#include <vector>
#include "simple_dict.h"
namespace baidu {
    int utf8_to_char(const std::string& sent, std::vector<std::string>& chars) {
        size_t len = sent.size();
        chars.clear();
        for (size_t i = 0; i < len;) {
            size_t beg = i;
            unsigned char p = (unsigned char) sent[i];
            if (p < 0x80) {
                if (p == ' ') {
                    ++i;
                }else {
                    while (i < len && p < 0x80 && p != ' ') {
                        p = (unsigned char) sent[++i];
                    }
                }
            }else if (p < 0xC0) {
                return -1;
            }else if (p < 0xE0) {
                i += 2;
            }else if (p < 0xF0) {
                i += 3;
            }else if (p < 0xF8) {
                i += 4;
            }else if (p < 0xFC) {
                i += 5;
            }else if (p < 0xFE) {
                i += 6;
            }else {
                return -1;
            }
            if (i > len) {
                return -1;
            }
            chars.push_back(sent.substr(beg, i - beg));
        }
        return 0;
    }
    class BasicTokenizer {
        public:
            BasicTokenizer(){}
            ~BasicTokenizer(){}
            int init(bool do_lower_case);
            int tokenize(const std::string &input_string, std::vector<std::string> &split_tokens);
            std::string clean_text(const std::string &input_text);
        private:
            bool _do_lower_case;
            std::string _punctuation;
    };
    class WordpieceTokenizer {
        public:
            int init(SimpleDict *p_simple_dict);
            WordpieceTokenizer(){}
            ~WordpieceTokenizer(){}
            int tokenize(const std::string &input_string, std::vector<std::string> &split_tokens);
        private:
            SimpleDict *_p_dict;
            std::string _unk_token;
            int _max_input_chars_per_word;
    };
    class FullTokenizer {
        public:
            FullTokenizer(){}
            ~FullTokenizer(){}
            int init(std::string &dict_file, bool do_lower_case);
            int tokenize(const std::string &input_string, std::vector<std::string> &split_tokens);
            int convert_tokens_to_ids(const std::vector<std::string>& tokens, std::vector<int> &ids);
            int convert_ids_to_tokens(const std::vector<int>& ids, std::vector<std::string> &tokens);
        private:
            bool _do_lower_case;
            SimpleDict _simple_dict;
            WordpieceTokenizer _wordpiece_tokenizer;
            BasicTokenizer _basic_tokenizer;
            //dict. basic_tokenizer, wordpiece_tokenizer
    };
}
#endif
