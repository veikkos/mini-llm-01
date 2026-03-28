// 06-mini-llm-cuda/native/bpe.cpp
#include "bpe.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cstdio>

BPE::BPE() {
    // Initialize base vocab: 256 byte tokens + <eos>
    buildVocab();
}

void BPE::buildVocab() {
    vocab_.clear();
    vocab_.resize(256 + 1); // 0-255 bytes + 256 eos
    for (int i = 0; i < 256; i++) {
        vocab_[i] = { (uint8_t)i };
    }
    vocab_[EOS_TOKEN] = {}; // <eos> has no byte representation

    // Add merged tokens
    for (int i = 0; i < (int)merges_.size(); i++) {
        auto& [left, right] = merges_[i];
        auto& lv = vocab_[left];
        auto& rv = vocab_[right];
        std::vector<uint8_t> merged;
        merged.insert(merged.end(), lv.begin(), lv.end());
        merged.insert(merged.end(), rv.begin(), rv.end());
        vocab_.push_back(std::move(merged));
    }

    // Build reverse lookup
    mergeRank_.clear();
    for (int i = 0; i < (int)merges_.size(); i++) {
        mergeRank_[merges_[i]] = 257 + i;
    }
}

void BPE::train(const uint8_t* data, size_t len, int targetVocabSize) {
    if (targetVocabSize <= 257) {
        throw std::runtime_error("targetVocabSize must be > 257 (256 bytes + <eos>)");
    }

    merges_.clear();

    // Start with byte-level tokens
    std::vector<int> ids(len);
    for (size_t i = 0; i < len; i++) {
        ids[i] = data[i];
    }

    int numMerges = targetVocabSize - 257; // 256 base + <eos> + merges

    for (int m = 0; m < numMerges; m++) {
        // Count adjacent pairs
        std::unordered_map<std::pair<int,int>, int, PairHash> pairCounts;
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            pairCounts[{ids[i], ids[i + 1]}]++;
        }

        if (pairCounts.empty()) break;

        // Find most frequent pair
        auto best = std::max_element(pairCounts.begin(), pairCounts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto bestPair = best->first;
        int newToken = 257 + m;

        merges_.push_back(bestPair);

        // Replace all occurrences of bestPair with newToken
        std::vector<int> newIds;
        newIds.reserve(ids.size());
        size_t i = 0;
        while (i < ids.size()) {
            if (i + 1 < ids.size() && ids[i] == bestPair.first && ids[i + 1] == bestPair.second) {
                newIds.push_back(newToken);
                i += 2;
            } else {
                newIds.push_back(ids[i]);
                i++;
            }
        }
        ids = std::move(newIds);

        if ((m + 1) % 100 == 0) {
            fprintf(stderr, "  merge %d/%d: (%d, %d) -> %d (corpus size: %zu)\n",
                    m + 1, numMerges, bestPair.first, bestPair.second, newToken, ids.size());
        }
    }

    buildVocab();
}

std::vector<int> BPE::encode(const std::string& text) const {
    if (text.empty()) return {};

    // Start with byte-level tokens
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        ids.push_back(c);
    }

    // Apply merges in priority order
    for (int i = 0; i < (int)merges_.size(); i++) {
        auto& [left, right] = merges_[i];
        int merged = 257 + i;

        std::vector<int> newIds;
        newIds.reserve(ids.size());
        size_t j = 0;
        while (j < ids.size()) {
            if (j + 1 < ids.size() && ids[j] == left && ids[j + 1] == right) {
                newIds.push_back(merged);
                j += 2;
            } else {
                newIds.push_back(ids[j]);
                j++;
            }
        }
        ids = std::move(newIds);
    }

    return ids;
}

std::string BPE::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int tok : tokens) {
        if (tok == EOS_TOKEN) break; // stop at <eos>
        if (tok < 0 || tok >= (int)vocab_.size()) continue;
        auto& bytes = vocab_[tok];
        result.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    return result;
}

void BPE::save(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

    fprintf(f, "{\n");
    fprintf(f, "  \"vocab_size\": %d,\n", vocabSize());
    fprintf(f, "  \"special_tokens\": { \"<eos>\": %d },\n", EOS_TOKEN);
    fprintf(f, "  \"merges\": [\n");
    for (size_t i = 0; i < merges_.size(); i++) {
        fprintf(f, "    [%d, %d]%s\n", merges_[i].first, merges_[i].second,
                i + 1 < merges_.size() ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

void BPE::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("Cannot open vocab file: " + path);

    // Read entire file
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string content(sz, '\0');
    fread(&content[0], 1, sz, f);
    fclose(f);

    // Parse merges array: find "merges" key, then read [int, int] pairs
    merges_.clear();
    size_t pos = content.find("\"merges\"");
    if (pos == std::string::npos) throw std::runtime_error("No merges key in vocab file");

    pos = content.find('[', pos); // opening [ of merges array
    if (pos == std::string::npos) throw std::runtime_error("Malformed merges in vocab file");
    pos++; // skip [

    while (pos < content.size()) {
        // Skip whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ',')) pos++;

        if (content[pos] == ']') break; // end of merges array

        if (content[pos] == '[') {
            pos++; // skip [
            int left = 0, right = 0;

            // Parse first int
            while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
                left = left * 10 + (content[pos] - '0');
                pos++;
            }
            // Skip comma and whitespace
            while (pos < content.size() && (content[pos] == ',' || content[pos] == ' ')) pos++;
            // Parse second int
            while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
                right = right * 10 + (content[pos] - '0');
                pos++;
            }
            // Skip to closing ]
            while (pos < content.size() && content[pos] != ']') pos++;
            pos++; // skip ]

            merges_.push_back({left, right});
        } else {
            pos++;
        }
    }

    buildVocab();
}
