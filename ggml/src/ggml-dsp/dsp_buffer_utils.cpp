#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>

#include <thread>
#include <queue>
#include <set>

#include "MTUtils.hpp"
#include "dsp_multicluster_buffer.h"

#include "dsp_utils.h"

typedef struct addr_range {
    void * ptr;
    size_t size;
    int cluster_id;
} addr_range_t;

static int main_cluster = 1;
static std::unordered_map<void *, addr_range_t> mtmalloc_set;
static std::set<transaction_t> work_queue;

int dsp_get_main_cluster() {
    return main_cluster;
}

transaction_t dsp_memcpy_async(void * dst, const void * src, size_t size) {
    // TODO: replace with thread pool
    transaction_t transc = (uint64_t)(void*)(new std::thread([=]() {
        memcpy(dst, src, size);
        mt_flush(dst, size);
    }));

    work_queue.insert(transc);

    return transc;
}

void dsp_memcpy(void * dst, const void * src, size_t size) {
    memcpy(dst, src, size);
    mt_flush(dst, size);
}

void dsp_transc_wait(uint64_t transaction_id) {
    auto it = work_queue.find(transaction_id);
    if(it != work_queue.end()) {
        std::thread * ptr = (std::thread *)(void *)transaction_id;
        ptr->join();
        work_queue.erase(it);
        delete ptr;
    }
    else { assert(false); }
}

void dsp_transc_wait_all() {
    for(transaction_t t : work_queue) {
        std::thread * ptr = (std::thread *)(void *)t;
        ptr->join();
        delete ptr; // Should not erase within iteration loop
    }
    work_queue.clear();
    mt_flush_all(); // should be useless, let's keep it for debug
}

void * dsp_malloc_on_cluster_at(size_t size, int cluster_id, void * fixed_addr) {
    void * data = mt_malloc(cluster_id, size, (unsigned long)fixed_addr);
    assert(data);
    assert((uint64_t)fixed_addr != 0x1000ul); // reserved as magic num in libmt

    addr_range_t addr_range;
    addr_range.cluster_id = cluster_id;
    addr_range.ptr = data;
    addr_range.size = size;

    mtmalloc_set[data] = addr_range;

    return data;
}

void * dsp_malloc_on_cluster(size_t size, int cluster_id) {
    return dsp_malloc_on_cluster_at(size, cluster_id, nullptr);
}

void dsp_free(void * ptr) {
    void * base_ptr = dsp_get_base_from_ptr(ptr);
    assert(base_ptr == ptr);
    
    auto it = mtmalloc_set.find(ptr);
    if(it != mtmalloc_set.end()) {
        mt_free(it->second.ptr, 0ul);
        mtmalloc_set.erase(it);
    }
}

int dsp_get_cluster_id_from_ptr(void * ptr) {
    auto it = mtmalloc_set.find(ptr);
    if(it != mtmalloc_set.end()) {
        return (*it).second.cluster_id;
    }

    for(const auto& p : mtmalloc_set) {
        if(p.second.ptr < ptr && ptr < ((uint8_t *)p.second.ptr + p.second.size)) {
            return p.second.cluster_id;
        }
    }

    return -1;
}

size_t dsp_get_alloc_size_from_ptr(void * ptr) {
    auto it = mtmalloc_set.find(ptr);
    if(it != mtmalloc_set.end()) {
        return (*it).second.size;
    }

    for(const auto& p : mtmalloc_set) {
        if(p.second.ptr < ptr && ptr < ((uint8_t *)p.second.ptr + p.second.size)) {
            return p.second.size;
        }
    }

    return -1;
}

void * dsp_get_base_from_ptr(void * ptr) {
    auto it = mtmalloc_set.find(ptr);
    if(it != mtmalloc_set.end()) {
        return (*it).second.ptr;
    }

    for(const auto& p : mtmalloc_set) {
        if(p.second.ptr < ptr && ptr < ((uint8_t *)p.second.ptr + p.second.size)) {
            return p.second.ptr;
        }
    }

    return nullptr;
}

transaction_t dsp_async_exc(std::function<void()> func) {
    transaction_t transc = (transaction_t)(void*)(new std::thread(func));
    work_queue.insert(transc);
    return transc;
}
