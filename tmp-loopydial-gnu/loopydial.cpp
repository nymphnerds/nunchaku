/*
 * Loopy Dial - disting NT utility plug-in
 * Two-way stepped Project Profile selector for Loopy Pro.
 */

#include <distingnt/api.h>
#include <distingnt/serialisation.h>
#include <new>
#include <cstring>

namespace
{

enum
{
    kParamProfile,
    kParamProfileCount,
    kParamHome,
    kParamMidiChannel,
    kParamMidiCC,
    kNumParams,
};

static const int kMaxProfiles = 32;
static const int kNameLength = 24;

static const _NT_parameter parameters[kNumParams] =
{
    { .name = "Profile",       .min = 1, .max = 2,            .def = 1,   .unit = kNT_unitHasStrings, .scaling = 0, .enumStrings = NULL },
    { .name = "Profile count", .min = 2, .max = kMaxProfiles, .def = 2,   .unit = kNT_unitNone,       .scaling = 0, .enumStrings = NULL },
    { .name = "Home",          .min = 1, .max = 2,            .def = 1,   .unit = kNT_unitHasStrings, .scaling = 0, .enumStrings = NULL },
    { .name = "MIDI channel",  .min = 1, .max = 16,           .def = 1,   .unit = kNT_unitNone,       .scaling = 0, .enumStrings = NULL },
    { .name = "MIDI CC",       .min = 0, .max = 127,          .def = 119, .unit = kNT_unitNone,       .scaling = 0, .enumStrings = NULL },
};

static const uint8_t pageProfile[] = { kParamProfile, kParamProfileCount, kParamHome };
static const uint8_t pageMidi[] = { kParamMidiChannel, kParamMidiCC };

static const _NT_parameterPage pages[] =
{
    { .name = "Profile", .numParams = ARRAY_SIZE(pageProfile), .params = pageProfile },
    { .name = "MIDI",    .numParams = ARRAY_SIZE(pageMidi),    .params = pageMidi },
};

static const _NT_parameterPages parameterPages =
{
    .numPages = ARRAY_SIZE(pages),
    .pages = pages,
};

struct LoopyDial : public _NT_algorithm
{
    LoopyDial() : editingName(false), nameCursor(0), ignoreFeedbackProfile(0) {}
    ~LoopyDial() {}

    _NT_parameter params[kNumParams];
    char names[kMaxProfiles][kNameLength];
    bool editingName;
    uint8_t nameCursor;
    int16_t ignoreFeedbackProfile;
};

static int clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int textLength(const char* s)
{
    return s ? (int)std::strlen(s) : 0;
}

static void copyText(char* dst, int capacity, const char* src)
{
    if (!dst || capacity <= 0) return;
    if (!src) { dst[0] = 0; return; }
    std::strncpy(dst, src, capacity - 1);
    dst[capacity - 1] = 0;
}

static void makeDefaultName(char* dst, int profile)
{
    copyText(dst, kNameLength, "Profile ");
    int len = textLength(dst);
    NT_intToString(dst + len, profile);
}

static void initialiseNames(LoopyDial* self)
{
    for (int i = 0; i < kMaxProfiles; ++i)
        makeDefaultName(self->names[i], i + 1);
    copyText(self->names[0], kNameLength, "WitchboardX");
    copyText(self->names[1], kNameLength, "FAC_Drumkit");
}

static int profileToMidi(int index, int count)
{
    count = clampi(count, 2, kMaxProfiles);
    index = clampi(index, 0, count - 1);
    const int den = count - 1;
    int value = (128 * index + den / 2) / den;
    return value > 127 ? 127 : value;
}

static int midiToProfile(int value, int count)
{
    value = clampi(value, 0, 127);
    count = clampi(count, 2, kMaxProfiles);
    int best = 0;
    int bestDistance = 1000;
    for (int i = 0; i < count; ++i)
    {
        int d = profileToMidi(i, count) - value;
        if (d < 0) d = -d;
        if (d < bestDistance)
        {
            bestDistance = d;
            best = i;
        }
    }
    return best + 1;
}

static void sendProfile(LoopyDial* self)
{
    const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
    const int profile = clampi(self->v[kParamProfile], 1, count);
    const uint8_t status = (uint8_t)(0xB0 | (clampi(self->v[kParamMidiChannel], 1, 16) - 1));
    const uint8_t cc = (uint8_t)clampi(self->v[kParamMidiCC], 0, 127);
    const uint8_t value = (uint8_t)profileToMidi(profile - 1, count);
    NT_sendMidi3ByteMessage(kNT_destinationUSB, status, cc, value);
}

static void updateProfileRanges(LoopyDial* self)
{
    const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
    const int algIndex = NT_algorithmIndex(self);
    if (algIndex < 0) return;

    self->params[kParamProfile].max = count;
    NT_updateParameterDefinition((uint32_t)algIndex, kParamProfile);

    self->params[kParamHome].max = count;
    NT_updateParameterDefinition((uint32_t)algIndex, kParamHome);

    if (self->v[kParamProfile] > count)
    {
        self->ignoreFeedbackProfile = count;
        NT_setParameterFromAudio((uint32_t)algIndex,
            kParamProfile + NT_parameterOffset(), count);
    }

    if (self->v[kParamHome] > count)
    {
        NT_setParameterFromAudio((uint32_t)algIndex,
            kParamHome + NT_parameterOffset(), count);
    }
}

static void calculateRequirements(_NT_algorithmRequirements& req, const int32_t*)
{
    req.numParameters = ARRAY_SIZE(parameters);
    req.sram = sizeof(LoopyDial);
    req.dram = 0;
    req.dtc = 0;
    req.itc = 0;
}

static _NT_algorithm* construct(const _NT_algorithmMemoryPtrs& ptrs,
                                const _NT_algorithmRequirements&,
                                const int32_t*)
{
    LoopyDial* self = new (ptrs.sram) LoopyDial();
    std::memcpy(self->params, parameters, sizeof(parameters));
    self->parameters = self->params;
    self->parameterPages = &parameterPages;
    initialiseNames(self);
    return self;
}

static void parameterChanged(_NT_algorithm* algorithm, int p)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    switch (p)
    {
    case kParamProfile:
    {
        const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
        const int selected = clampi(self->v[kParamProfile], 1, count);
        if (self->ignoreFeedbackProfile == selected)
        {
            self->ignoreFeedbackProfile = 0;
            return;
        }
        sendProfile(self);
    }
        break;

    case kParamProfileCount:
        updateProfileRanges(self);
        sendProfile(self);
        break;

    default:
        break;
    }
}

static void step(_NT_algorithm*, float*, int) {}

static void midiMessage(_NT_algorithm* algorithm, uint8_t b0, uint8_t b1, uint8_t b2)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    const uint8_t expectedStatus = (uint8_t)(0xB0 | (clampi(self->v[kParamMidiChannel], 1, 16) - 1));
    const uint8_t expectedCC = (uint8_t)clampi(self->v[kParamMidiCC], 0, 127);
    if (b0 != expectedStatus || b1 != expectedCC || b2 > 127) return;

    const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
    const int profile = midiToProfile(b2, count);
    if (profile == self->v[kParamProfile]) return;

    const int algIndex = NT_algorithmIndex(self);
    if (algIndex < 0) return;
    self->ignoreFeedbackProfile = profile;
    NT_setParameterFromAudio((uint32_t)algIndex,
        kParamProfile + NT_parameterOffset(), profile);
}

static int parameterString(_NT_algorithm* algorithm, int p, int v, char* buff)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    if (p != kParamProfile && p != kParamHome) return 0;
    const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
    const int profile = clampi(v, 1, count);
    copyText(buff, kNT_parameterStringSize, self->names[profile - 1]);
    return textLength(buff);
}

static uint32_t hasCustomUi(_NT_algorithm* algorithm)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    if (self->editingName)
        return kNT_button2 | kNT_encoderL | kNT_encoderR;
    return kNT_button2 | kNT_encoderButtonL;
}

static bool pressed(uint16_t bit, const _NT_uiData& data)
{
    return (data.controls & bit) && !(data.lastButtons & bit);
}

static void customUi(_NT_algorithm* algorithm, const _NT_uiData& data)
{
    LoopyDial* self = (LoopyDial*)algorithm;

    if (pressed(kNT_button2, data))
    {
        self->editingName = !self->editingName;
        self->nameCursor = 0;
        return;
    }

    if (!self->editingName)
    {
        if (pressed(kNT_encoderButtonL, data))
        {
            const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
            const int home = clampi(self->v[kParamHome], 1, count);
            const int algIndex = NT_algorithmIndex(self);
            if (algIndex >= 0)
                NT_setParameterFromUi((uint32_t)algIndex,
                    kParamProfile + NT_parameterOffset(), home);
        }
        return;
    }

    const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
    const int profile = clampi(self->v[kParamProfile], 1, count);
    char* name = self->names[profile - 1];

    if (data.encoders[1])
    {
        int cursor = (int)self->nameCursor + (data.encoders[1] > 0 ? 1 : -1);
        self->nameCursor = (uint8_t)clampi(cursor, 0, kNameLength - 2);
    }

    if (data.encoders[0])
    {
        int len = textLength(name);
        while (len <= (int)self->nameCursor && len < kNameLength - 1)
        {
            name[len++] = ' ';
            name[len] = 0;
        }
        int c = (unsigned char)name[self->nameCursor];
        if (c < 32 || c > 126) c = 32;
        c += data.encoders[0] > 0 ? 1 : -1;
        if (c > 126) c = 32;
        if (c < 32) c = 126;
        name[self->nameCursor] = (char)c;
    }
}

static bool draw(_NT_algorithm* algorithm)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    const int count = clampi(self->v[kParamProfileCount], 2, kMaxProfiles);
    const int profile = clampi(self->v[kParamProfile], 1, count);

    char top[24];
    copyText(top, sizeof(top), "Profile ");
    int len = textLength(top);
    len += NT_intToString(top + len, profile);
    top[len++] = '/';
    len += NT_intToString(top + len, count);
    top[len] = 0;

    NT_drawText(6, 27, top);
    NT_drawText(6, 49, self->names[profile - 1]);
    if (self->editingName)
        NT_drawText(6, 61, "B2 done  L char  R cursor", 8);
    else
        NT_drawText(6, 61, "L push Home  B2 rename", 8);
    return false;
}

static void serialise(_NT_algorithm* algorithm, _NT_jsonStream& stream)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    stream.addMemberName("profileNames");
    stream.openArray();
    for (int i = 0; i < kMaxProfiles; ++i)
        stream.addString(self->names[i]);
    stream.closeArray();
}

static bool deserialise(_NT_algorithm* algorithm, _NT_jsonParse& parse)
{
    LoopyDial* self = (LoopyDial*)algorithm;
    int numMembers = 0;
    if (!parse.numberOfObjectMembers(numMembers)) return false;

    for (int i = 0; i < numMembers; ++i)
    {
        if (parse.matchName("profileNames"))
        {
            int numNames = 0;
            if (!parse.numberOfArrayElements(numNames)) return false;
            for (int n = 0; n < numNames; ++n)
            {
                const char* src = NULL;
                if (!parse.string(src)) return false;
                if (n < kMaxProfiles)
                    copyText(self->names[n], kNameLength, src);
            }
        }
        else if (!parse.skipMember())
            return false;
    }

    self->editingName = false;
    self->nameCursor = 0;
    self->ignoreFeedbackProfile = 0;
    return true;
}

static const _NT_factory factory =
{
    .guid = NT_MULTICHAR('L', 'p', 'D', 'l'),
    .name = "Loopy Dial",
    .description = "Loopy Pro profile selector",
    .numSpecifications = 0,
    .specifications = NULL,
    .calculateStaticRequirements = NULL,
    .initialise = NULL,
    .calculateRequirements = calculateRequirements,
    .construct = construct,
    .parameterChanged = parameterChanged,
    .step = step,
    .draw = draw,
    .midiRealtime = NULL,
    .midiMessage = midiMessage,
    .tags = kNT_tagUtility,
    .hasCustomUi = hasCustomUi,
    .customUi = customUi,
    .setupUi = NULL,
    .serialise = serialise,
    .deserialise = deserialise,
    .midiSysEx = NULL,
    .parameterUiPrefix = NULL,
    .parameterString = parameterString,
};

} // namespace

extern "C" uintptr_t pluginEntry(_NT_selector selector, uint32_t data)
{
    switch (selector)
    {
    case kNT_selector_version:
        return kNT_apiVersionCurrent;
    case kNT_selector_numFactories:
        return 1;
    case kNT_selector_factoryInfo:
        return (uintptr_t)((data == 0) ? &factory : NULL);
    }
    return 0;
}
