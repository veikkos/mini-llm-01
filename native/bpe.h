// 06-mini-llm-cuda/native/bpe.h
#ifndef BPE_H
#define BPE_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <utility>

struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<int64_t>()(((int64_t)p.first << 32) | (uint32_t)p.second);
    }
};

class BPE {
public:
    static constexpr int EOS_TOKEN = 256;

    BPE();

    // Train: learn merges from raw bytes
    void train(const uint8_t* data, size_t len, int targetVocabSize);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text) const;

    // Decode token IDs back to text
    std::string decode(const std::vector<int>& tokens) const;

    // Persistence
    void save(const std::string& path) const;
    void load(const std::string& path);

    int vocabSize() const { return (int)vocab_.size(); }

private:
    // Ordered merge rules: merges_[i] = {left, right} -> token (257 + i)
    std::vector<std::pair<int,int>> merges_;

    // vocab_[tokenId] = byte string for that token
    std::vector<std::vector<uint8_t>> vocab_;

    // Reverse lookup: (left, right) -> merged token id
    std::unordered_map<std::pair<int,int>, int, PairHash> mergeRank_;

    void buildVocab();
};

#endif
