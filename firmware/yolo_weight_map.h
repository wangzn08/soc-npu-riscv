#ifndef YOLO_WEIGHT_MAP_H
#define YOLO_WEIGHT_MAP_H
#include <stdint.h>

#define YOLO_WGT_DDR_BASE 0x40800000u
#define YOLO_WGT_TOTAL_WORDS 196753u
// per-conv: {ddr_word_off, wgt_words, oc, ic, kh, kw}
typedef struct { uint32_t off, words, oc, ic, kh, kw; } yolo_wgt_ent_t;
static const yolo_wgt_ent_t yolo_wgt_map[64] = {
    {0u, 144u, 16u, 3u, 3u, 3u},  // conv0
    {144u, 288u, 32u, 16u, 3u, 3u},  // conv1
    {432u, 64u, 32u, 32u, 1u, 1u},  // conv2
    {496u, 144u, 16u, 16u, 3u, 3u},  // conv3
    {640u, 144u, 16u, 16u, 3u, 3u},  // conv4
    {784u, 96u, 32u, 48u, 1u, 1u},  // conv5
    {880u, 1152u, 64u, 32u, 3u, 3u},  // conv6
    {2032u, 256u, 64u, 64u, 1u, 1u},  // conv7
    {2288u, 576u, 32u, 32u, 3u, 3u},  // conv8
    {2864u, 576u, 32u, 32u, 3u, 3u},  // conv9
    {3440u, 576u, 32u, 32u, 3u, 3u},  // conv10
    {4016u, 576u, 32u, 32u, 3u, 3u},  // conv11
    {4592u, 512u, 64u, 128u, 1u, 1u},  // conv12
    {5104u, 4608u, 128u, 64u, 3u, 3u},  // conv13
    {9712u, 1024u, 128u, 128u, 1u, 1u},  // conv14
    {10736u, 2304u, 64u, 64u, 3u, 3u},  // conv15
    {13040u, 2304u, 64u, 64u, 3u, 3u},  // conv16
    {15344u, 2304u, 64u, 64u, 3u, 3u},  // conv17
    {17648u, 2304u, 64u, 64u, 3u, 3u},  // conv18
    {19952u, 2048u, 128u, 256u, 1u, 1u},  // conv19
    {22000u, 18432u, 256u, 128u, 3u, 3u},  // conv20
    {40432u, 4096u, 256u, 256u, 1u, 1u},  // conv21
    {44528u, 9216u, 128u, 128u, 3u, 3u},  // conv22
    {53744u, 9216u, 128u, 128u, 3u, 3u},  // conv23
    {62960u, 6144u, 256u, 384u, 1u, 1u},  // conv24
    {69104u, 2048u, 128u, 256u, 1u, 1u},  // conv25
    {71152u, 8192u, 256u, 512u, 1u, 1u},  // conv26
    {79344u, 3072u, 128u, 384u, 1u, 1u},  // conv27
    {82416u, 2304u, 64u, 64u, 3u, 3u},  // conv28
    {84720u, 2304u, 64u, 64u, 3u, 3u},  // conv29
    {87024u, 1536u, 128u, 192u, 1u, 1u},  // conv30
    {88560u, 768u, 64u, 192u, 1u, 1u},  // conv31
    {89328u, 576u, 32u, 32u, 3u, 3u},  // conv32
    {89904u, 576u, 32u, 32u, 3u, 3u},  // conv33
    {90480u, 384u, 64u, 96u, 1u, 1u},  // conv34
    {90864u, 2304u, 64u, 64u, 3u, 3u},  // conv35
    {93168u, 2304u, 64u, 64u, 3u, 3u},  // conv36
    {95472u, 2880u, 80u, 64u, 3u, 3u},  // conv37
    {98352u, 2304u, 64u, 64u, 3u, 3u},  // conv38
    {100656u, 3600u, 80u, 80u, 3u, 3u},  // conv39
    {104256u, 1536u, 128u, 192u, 1u, 1u},  // conv40
    {105792u, 256u, 64u, 64u, 1u, 1u},  // conv41
    {106048u, 400u, 80u, 80u, 1u, 1u},  // conv42
    {106448u, 2304u, 64u, 64u, 3u, 3u},  // conv43
    {108752u, 2304u, 64u, 64u, 3u, 3u},  // conv44
    {111056u, 1536u, 128u, 192u, 1u, 1u},  // conv45
    {112592u, 9216u, 128u, 128u, 3u, 3u},  // conv46
    {121808u, 4608u, 64u, 128u, 3u, 3u},  // conv47
    {126416u, 5760u, 80u, 128u, 3u, 3u},  // conv48
    {132176u, 2304u, 64u, 64u, 3u, 3u},  // conv49
    {134480u, 3600u, 80u, 80u, 3u, 3u},  // conv50
    {138080u, 6144u, 256u, 384u, 1u, 1u},  // conv51
    {144224u, 256u, 64u, 64u, 1u, 1u},  // conv52
    {144480u, 400u, 80u, 80u, 1u, 1u},  // conv53
    {144880u, 9216u, 128u, 128u, 3u, 3u},  // conv54
    {154096u, 9216u, 128u, 128u, 3u, 3u},  // conv55
    {163312u, 6144u, 256u, 384u, 1u, 1u},  // conv56
    {169456u, 9216u, 64u, 256u, 3u, 3u},  // conv57
    {178672u, 11520u, 80u, 256u, 3u, 3u},  // conv58
    {190192u, 2304u, 64u, 64u, 3u, 3u},  // conv59
    {192496u, 3600u, 80u, 80u, 3u, 3u},  // conv60
    {196096u, 256u, 64u, 64u, 1u, 1u},  // conv61
    {196352u, 400u, 80u, 80u, 1u, 1u},  // conv62
    {196752u, 1u, 1u, 16u, 1u, 1u},  // conv63
};
#endif
