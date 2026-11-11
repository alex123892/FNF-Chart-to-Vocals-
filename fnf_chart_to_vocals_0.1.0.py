import json
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
import importlib.util

from pathlib import Path
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "FNF Chart to Vocals Generator"

DIRECTION_PITCHES = {
    0: 60,
    1: 62,
    2: 64,
    3: 65
}

SAMPLE_RATE = 22050
FFT_WINDOW_SIZE = 4096
FFT_SIZE = 16384
TICKS_PER_BEAT = 480

VOICE_MIN_PITCH = 48
VOICE_MAX_PITCH = 84


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def get_vslice_difficulties(data):
    notes = data.get("notes")

    if isinstance(notes, dict):
        return [
            name for name, items in notes.items()
            if isinstance(items, list)
        ]

    return []


def get_opponent_character(metadata_path):
    if not metadata_path:
        return "opponent"

    try:
        metadata = load_json(metadata_path)

        opponent = metadata.get(
            "playData", {}
        ).get(
            "characters", {}
        ).get("opponent")

        if isinstance(opponent, str) and opponent.strip():
            safe_name = "".join(
                ch for ch in opponent.strip()
                if ch.isalnum() or ch in "-_"
            )

            return safe_name or "opponent"

    except Exception:
        pass

    return "opponent"


def read_vslice_chart(data, difficulty):
    notes_by_difficulty = data.get("notes")

    if not isinstance(notes_by_difficulty, dict):
        raise ValueError(
            "Elegí el archivo *-chart.json de V-Slice."
        )

    difficulties = get_vslice_difficulties(data)

    if not difficulties:
        raise ValueError(
            "El chart no contiene dificultades."
        )

    if not difficulty or difficulty not in notes_by_difficulty:
        difficulty = difficulties[0]

    bf_notes = []
    opponent_notes = []

    for note in notes_by_difficulty[difficulty]:
        if not isinstance(note, dict):
            continue

        try:
            time_ms = float(note["t"])
            raw_data = int(note["d"])
            sustain_ms = float(note.get("l", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue

        if raw_data < 0:
            continue

        note_data = raw_data % 8
        lane = note_data % 4

        entry = (
            time_ms,
            lane,
            max(0.0, sustain_ms)
        )

        if note_data < 4:
            bf_notes.append(entry)
        else:
            opponent_notes.append(entry)

    return bf_notes, opponent_notes, difficulties


def read_psych_chart(data):
    song = data.get("song", data)
    sections = song.get("notes")

    if isinstance(sections, list):
        bf_notes = []
        opponent_notes = []
        found_sections = False

        for section in sections:
            if not isinstance(section, dict):
                continue

            section_notes = section.get("sectionNotes", [])

            if not isinstance(section_notes, list):
                continue

            found_sections = True
            must_hit = bool(section.get("mustHitSection", False))

            for note in section_notes:
                if not isinstance(note, (list, tuple)):
                    continue

                if len(note) < 2:
                    continue

                try:
                    time_ms = float(note[0])
                    lane = int(note[1])

                    sustain_ms = (
                        float(note[2])
                        if len(note) > 2
                        else 0.0
                    )

                except (TypeError, ValueError):
                    continue

                if lane < 0:
                    continue

                belongs_to_bf = (
                    must_hit if lane < 4 else not must_hit
                )

                entry = (
                    time_ms,
                    lane % 4,
                    max(0.0, sustain_ms)
                )

                if belongs_to_bf:
                    bf_notes.append(entry)
                else:
                    opponent_notes.append(entry)

        if found_sections:
            return bf_notes, opponent_notes

    raise ValueError(
        "No reconocí el formato del chart JSON."
    )


def read_chart(path, difficulty):
    data = load_json(path)

    if isinstance(data.get("notes"), dict):
        return read_vslice_chart(data, difficulty)

    bf, opponent = read_psych_chart(data)

    return bf, opponent, ["Psych / Legacy"]


def find_tool(name):
    return shutil.which(name)


def run_command(args):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        detail = (
            result.stderr
            or result.stdout
            or "Error sin detalles"
        ).strip()

        raise RuntimeError(detail[-3500:])

    return result


def safe_ogg_filename(value, fallback):
    value = (value or "").strip()

    if not value:
        value = fallback

    value = Path(value).name

    for character in '<>:"/\\|?*':
        value = value.replace(character, "_")

    if value.lower().endswith(".ogg"):
        value = value[:-4]
    else:
        value = Path(value).stem

    value = value.strip(" .")

    if not value:
        value = Path(fallback).stem

    return value + ".ogg"


def decode_instrumental(ffmpeg, audio_path):
    import numpy as np

    result = subprocess.run(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(audio_path),
            "-f", "s16le",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "pipe:1"
        ],
        capture_output=True
    )

    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode(
            "utf-8", "replace"
        )[-2000:]

        raise RuntimeError(
            "No pude analizar la instrumental.\n" + detail
        )

    usable = len(result.stdout) - len(result.stdout) % 2

    samples = np.frombuffer(
        result.stdout[:usable],
        dtype="<i2"
    ).astype(np.float32)

    samples /= 32768.0

    return samples


def midi_to_frequency(pitch):
    return 440.0 * (
        2.0 ** ((pitch - 69) / 12.0)
    )


def nearest_voice_pitch(detected_pitch, base_pitch):
    candidates = []

    for octave in range(-5, 6):
        pitch = detected_pitch + octave * 12

        if VOICE_MIN_PITCH <= pitch <= VOICE_MAX_PITCH:
            candidates.append(pitch)

    if not candidates:
        return max(
            VOICE_MIN_PITCH,
            min(VOICE_MAX_PITCH, detected_pitch)
        )

    return min(
        candidates,
        key=lambda pitch: abs(pitch - base_pitch)
    )


def detect_pitch_at_time(samples, time_ms, base_pitch, cache):
    import numpy as np

    cache_key = (
        round(time_ms / 5.0),
        base_pitch
    )

    if cache_key in cache:
        return cache[cache_key]

    center = int(
        round(time_ms * SAMPLE_RATE / 1000.0)
    )

    start = center - FFT_WINDOW_SIZE // 2
    end = start + FFT_WINDOW_SIZE

    source_start = max(0, start)
    source_end = min(len(samples), end)

    segment = samples[source_start:source_end]

    left_padding = max(0, -start)

    right_padding = max(
        0,
        FFT_WINDOW_SIZE - left_padding - len(segment)
    )

    segment = np.pad(
        segment,
        (left_padding, right_padding)
    )

    segment = segment[:FFT_WINDOW_SIZE]

    if len(segment) != FFT_WINDOW_SIZE:
        cache[cache_key] = None
        return None

    rms = float(
        np.sqrt(np.mean(segment * segment))
    )

    if rms < 0.002:
        cache[cache_key] = None
        return None

    window = np.hanning(FFT_WINDOW_SIZE)

    spectrum = np.abs(
        np.fft.rfft(
            segment * window,
            n=FFT_SIZE
        )
    )

    bin_width = SAMPLE_RATE / FFT_SIZE

    max_spectrum = float(
        np.max(spectrum)
    )

    if max_spectrum <= 0:
        cache[cache_key] = None
        return None

    def get_peak(frequency):
        if frequency <= 0 or frequency >= SAMPLE_RATE / 2:
            return 0.0

        center_bin = int(
            round(frequency / bin_width)
        )

        radius = max(
            1,
            int(round(frequency * 0.012 / bin_width))
        )

        low = max(1, center_bin - radius)

        high = min(
            len(spectrum) - 1,
            center_bin + radius
        )

        if low > high:
            return 0.0

        return float(
            np.max(spectrum[low:high + 1])
        )

    best_pitch = None
    best_score = 0.0

    harmonic_weights = [
        1.0,
        0.65,
        0.45,
        0.30,
        0.20,
        0.14
    ]

    for midi_pitch in range(36, 91):
        fundamental = midi_to_frequency(midi_pitch)

        if not 60 <= fundamental <= 1400:
            continue

        energies = []

        for harmonic in range(1, 7):
            frequency = fundamental * harmonic

            if frequency >= SAMPLE_RATE / 2:
                break

            energies.append(
                get_peak(frequency)
            )

        if not energies:
            continue

        fundamental_energy = energies[0]

        score = sum(
            energies[index] * harmonic_weights[index]
            for index in range(len(energies))
        )

        harmonic_max = max(
            energies[1:],
            default=0.0
        )

        if harmonic_max > 0:
            clarity = (
                fundamental_energy /
                (harmonic_max + 0.000001)
            )

            score *= (
                0.75 + 0.25 * min(1.0, clarity)
            )

        subharmonic = get_peak(
            fundamental / 2.0
        )

        if (
            subharmonic > fundamental_energy * 1.35
            and subharmonic > max_spectrum * 0.08
        ):
            score *= 0.82

        if score > best_score:
            best_score = score
            best_pitch = midi_pitch

    if (
        best_pitch is None
        or best_score < max_spectrum * 0.14
    ):
        result = None
    else:
        result = nearest_voice_pitch(
            best_pitch,
            base_pitch
        )

    cache[cache_key] = result

    return result


def match_notes_to_instrument(notes, samples, strength):
    strength = max(
        0.0,
        min(100.0, float(strength))
    ) / 100.0

    cache = {}

    matched = []
    detected_count = 0
    fallback_count = 0

    for time_ms, lane, sustain_ms in notes:
        base_pitch = DIRECTION_PITCHES.get(
            lane,
            60
        )

        final_pitch = base_pitch

        detected_pitch = detect_pitch_at_time(
            samples,
            time_ms,
            base_pitch,
            cache
        )

        if detected_pitch is not None and strength > 0:
            difference = (
                detected_pitch - base_pitch
            )

            final_pitch = round(
                base_pitch + difference * strength
            )

            detected_count += 1

        else:
            fallback_count += 1

        matched.append((
            time_ms,
            lane,
            sustain_ms,
            max(0, min(127, final_pitch))
        ))

    return matched, detected_count, fallback_count


def make_midi(
    notes,
    output_path,
    extra_duration_ms=0,
    monophonic=True
):
    try:
        import mido
    except ImportError:
        raise RuntimeError(
            "Falta mido. Instalalo con: py -m pip install mido"
        )

    midi = mido.MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )

    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(
        mido.MetaMessage(
            "set_tempo",
            tempo=500000,
            time=0
        )
    )

    prepared = []

    for index, note in enumerate(notes):
        time_ms, lane, sustain_ms = note[:3]

        pitch = (
            int(note[3])
            if len(note) >= 4
            else DIRECTION_PITCHES.get(lane, 60)
        )

        start_tick = max(
            0,
            round(time_ms * TICKS_PER_BEAT * 2 / 1000)
        )

        duration_ms = (
            max(80.0, sustain_ms)
            + max(0, int(extra_duration_ms))
        )

        duration_tick = max(
            1,
            round(duration_ms * TICKS_PER_BEAT * 2 / 1000)
        )

        prepared.append((
            start_tick,
            start_tick + duration_tick,
            pitch,
            index
        ))

    prepared.sort(
        key=lambda item: (item[0], item[3])
    )

    if monophonic:
        unique_starts = {}

        for item in prepared:
            unique_starts[item[0]] = item

        prepared = sorted(
            unique_starts.values(),
            key=lambda item: (item[0], item[3])
        )

    events = []

    for index, (start_tick, end_tick, pitch, original_index) in enumerate(prepared):
        if monophonic and index + 1 < len(prepared):
            next_start = prepared[index + 1][0]

            if next_start < end_tick:
                end_tick = next_start

        if end_tick <= start_tick:
            continue

        events.append((
            start_tick,
            1,
            pitch
        ))

        events.append((
            end_tick,
            0,
            pitch
        ))

    events.sort(
        key=lambda event: (event[0], event[1])
    )

    last_tick = 0

    for tick, is_on, pitch in events:
        delta = max(
            0,
            tick - last_tick
        )

        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=pitch,
                velocity=100 if is_on else 0,
                channel=0,
                time=delta
            )
        )

        last_tick = tick

    track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0
        )
    )

    midi.save(str(output_path))


def render_voice(
    fluidsynth,
    ffmpeg,
    notes,
    sf2,
    gain_db,
    tag,
    temp_folder,
    extra_duration_ms,
    monophonic
):
    if not notes:
        return None

    temp_folder = Path(temp_folder)

    midi_path = temp_folder / f"{tag}.mid"
    wav_path = temp_folder / f"{tag}.wav"
    ogg_path = temp_folder / f"{tag}.ogg"

    make_midi(
        notes,
        midi_path,
        extra_duration_ms,
        monophonic
    )

    run_command([
        fluidsynth,
        "-ni",
        "-F", str(wav_path),
        "-r", "44100",
        str(sf2),
        str(midi_path)
    ])

    run_command([
        ffmpeg,
        "-y",
        "-i", str(wav_path),
        "-af",
        (
            f"volume={gain_db}dB,"
            "alimiter=limit=0.95:level=false"
        ),
        "-c:a", "libvorbis",
        "-q:a", "5",
        str(ogg_path)
    ])

    return ogg_path


def mix_audio_files(ffmpeg, sources, output_path):
    if not sources:
        raise ValueError(
            "No seleccionaste ninguna pista."
        )

    command = [ffmpeg, "-y"]

    for audio_path in sources:
        command.extend([
            "-i",
            str(audio_path)
        ])

    filter_parts = []
    labels = []

    for index in range(len(sources)):
        label = f"a{index}"

        filter_parts.append(
            f"[{index}:a]"
            "aresample=44100,"
            "aformat=sample_fmts=fltp:"
            "channel_layouts=stereo"
            f"[{label}]"
        )

        labels.append(f"[{label}]")

    if len(sources) == 1:
        filter_parts.append(
            f"{labels[0]}anull[mix]"
        )
    else:
        filter_parts.append(
            "".join(labels)
            +
            f"amix=inputs={len(sources)}:"
            "duration=longest:"
            "dropout_transition=0:"
            "normalize=0,"
            "alimiter=limit=0.95:level=false"
            "[mix]"
        )

    command.extend([
        "-filter_complex",
        ";".join(filter_parts),
        "-map", "[mix]",
        "-c:a", "libvorbis",
        "-q:a", "5",
        str(output_path)
    ])

    run_command(command)


def build_selected_audio(config, ffmpeg, fluidsynth, temp_folder):
    sources = []
    generated_voices = {}

    detected_total = 0
    fallback_total = 0

    needs_bf = config["bf_enabled"]
    needs_opponent = config["opponent_enabled"]

    needs_analysis = (
        config["sync_percent"] > 0
        and (needs_bf or needs_opponent)
    )

    if config["instrumental_enabled"]:
        sources.append(
            config["instrumental"]
        )

    if needs_bf or needs_opponent:
        bf_notes, opponent_notes, _ = read_chart(
            config["chart"],
            config["difficulty"]
        )

        if needs_analysis:
            samples = decode_instrumental(
                ffmpeg,
                config["instrumental"]
            )

            if needs_bf:
                bf_notes, detected, fallback = match_notes_to_instrument(
                    bf_notes,
                    samples,
                    config["sync_percent"]
                )

                detected_total += detected
                fallback_total += fallback

            if needs_opponent:
                opponent_notes, detected, fallback = match_notes_to_instrument(
                    opponent_notes,
                    samples,
                    config["sync_percent"]
                )

                detected_total += detected
                fallback_total += fallback

        if needs_bf:
            if not bf_notes:
                raise ValueError(
                    "El chart no contiene notas de BF."
                )

            bf_path = render_voice(
                fluidsynth,
                ffmpeg,
                bf_notes,
                config["bf_sf2"],
                config["bf_gain"],
                "bf",
                temp_folder,
                config["extra_duration_ms"],
                config["monophonic"]
            )

            generated_voices["bf"] = bf_path
            sources.append(bf_path)

        if needs_opponent:
            if not opponent_notes:
                raise ValueError(
                    "El chart no contiene notas del oponente."
                )

            opponent_path = render_voice(
                fluidsynth,
                ffmpeg,
                opponent_notes,
                config["opponent_sf2"],
                config["opponent_gain"],
                "opponent",
                temp_folder,
                config["extra_duration_ms"],
                config["monophonic"]
            )

            generated_voices["opponent"] = opponent_path
            sources.append(opponent_path)

    return (
        sources,
        generated_voices,
        detected_total,
        fallback_total
    )


class App:
    def __init__(self, root):
        self.root = root

        self.root.title(APP_TITLE)
        self.root.geometry("940x900")
        self.root.minsize(820, 760)

        self.chart = tk.StringVar()
        self.instrumental = tk.StringVar()
        self.metadata = tk.StringVar()

        self.bf_sf2 = tk.StringVar()
        self.opponent_sf2 = tk.StringVar()

        self.output = tk.StringVar(
            value=str(
                Path.home() / "Music" / "FNF_Vocals"
            )
        )

        self.difficulty = tk.StringVar()

        self.sync_percent = tk.IntVar(value=70)
        self.sync_label = tk.StringVar(value="70%")

        self.bf_gain = tk.IntVar(value=8)
        self.opponent_gain = tk.IntVar(value=8)

        self.bf_gain_label = tk.StringVar(value="+8 dB")
        self.opponent_gain_label = tk.StringVar(value="+8 dB")

        self.extend_notes = tk.BooleanVar(value=False)
        self.extra_duration_ms = tk.IntVar(value=80)

        self.monophonic = tk.BooleanVar(value=True)

        self.instrumental_enabled = tk.BooleanVar(value=True)
        self.bf_enabled = tk.BooleanVar(value=False)
        self.opponent_enabled = tk.BooleanVar(value=False)

        self.bf_export_name = tk.StringVar(
            value="Voices-bf.ogg"
        )

        self.opponent_export_name = tk.StringVar(
            value="Voices-opponent.ogg"
        )

        self.mix_export_name = tk.StringVar(
            value="Preview-Mix.ogg"
        )

        self.status = tk.StringVar(
            value="Seleccioná las pistas que querés escuchar."
        )

        self.requirements = tk.StringVar()

        self.busy = False

        self.preview_process = None
        self.preview_thread = None
        self.preview_stop_event = None

        self.preview_lock = threading.Lock()

        self.build()
        self.connect_validation()
        self.update_note_controls()
        self.update_requirements()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    def build(self):
        outer = ttk.Frame(
            self.root,
            padding=16
        )

        outer.pack(
            fill="both",
            expand=True
        )

        canvas = tk.Canvas(
            outer,
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=canvas.yview
        )

        content = ttk.Frame(canvas)

        content.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        window = canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw"
        )

        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(
                window,
                width=event.width
            )
        )

        canvas.configure(
            yscrollcommand=scrollbar.set
        )

        canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.content = content

        ttk.Label(
            content,
            text=APP_TITLE,
            font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")

        ttk.Label(
            content,
            text="V-Slice / Psych Engine · Automatic Pitch Matching"
        ).pack(
            anchor="w",
            pady=(2, 12)
        )

        self.file_row(
            content,
            "Chart JSON",
            self.chart,
            self.pick_chart
        )

        self.file_row(
            content,
            "Instrumental",
            self.instrumental,
            self.pick_instrumental
        )

        self.file_row(
            content,
            "Metadata JSON (opcional)",
            self.metadata,
            self.pick_metadata
        )

        self.file_row(
            content,
            "Boyfriend SF2",
            self.bf_sf2,
            self.pick_bf
        )

        self.file_row(
            content,
            "Opponent SF2",
            self.opponent_sf2,
            self.pick_opponent
        )

        self.file_row(
            content,
            "Carpeta de exportación",
            self.output,
            self.pick_output
        )

        diff_row = ttk.Frame(content)
        diff_row.pack(fill="x", pady=5)

        ttk.Label(
            diff_row,
            text="Dificultad",
            width=25
        ).pack(side="left")

        self.difficulty_box = ttk.Combobox(
            diff_row,
            textvariable=self.difficulty,
            values=[],
            state="readonly",
            width=35
        )

        self.difficulty_box.pack(
            side="left",
            padx=7
        )

        self.difficulty_box.set(
            "Seleccioná un chart"
        )

        analysis = ttk.LabelFrame(
            content,
            text="AUDIO ANALYSIS",
            padding=10
        )

        analysis.pack(
            fill="x",
            pady=8
        )

        sync_row = ttk.Frame(analysis)
        sync_row.pack(
            fill="x"
        )

        ttk.Label(
            sync_row,
            text="Coincidencia tonal",
            width=25
        ).pack(side="left")

        ttk.Scale(
            sync_row,
            from_=0,
            to=100,
            variable=self.sync_percent,
            orient="horizontal",
            command=self.update_sync_label
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=7
        )

        ttk.Label(
            sync_row,
            textvariable=self.sync_label,
            width=6
        ).pack(side="left")

        ttk.Label(
            analysis,
            text=(
                "0% = tono original · 70% = ajuste parcial · "
                "100% = ajuste tonal completo. "
                "Los tiempos del chart permanecen intactos."
            ),
            foreground="#666",
            wraplength=850
        ).pack(
            anchor="w",
            pady=(5, 0)
        )

        volumes = ttk.LabelFrame(
            content,
            text="VOICE VOLUME",
            padding=10
        )

        volumes.pack(
            fill="x",
            pady=8
        )

        self.volume_row(
            volumes,
            "Boyfriend Voices",
            self.bf_gain,
            self.bf_gain_label
        )

        self.volume_row(
            volumes,
            "Opponent Voices",
            self.opponent_gain,
            self.opponent_gain_label
        )

        behavior = ttk.LabelFrame(
            content,
            text="NOTE BEHAVIOR",
            padding=10
        )

        behavior.pack(
            fill="x",
            pady=8
        )

        ttk.Checkbutton(
            behavior,
            text="Alargar la duración de las notas",
            variable=self.extend_notes,
            command=self.update_note_controls
        ).pack(anchor="w")

        duration_row = ttk.Frame(behavior)
        duration_row.pack(
            fill="x",
            pady=(5, 8)
        )

        ttk.Label(
            duration_row,
            text="Tiempo adicional:",
            width=25
        ).pack(side="left")

        self.duration_box = ttk.Spinbox(
            duration_row,
            from_=0,
            to=1000,
            increment=10,
            textvariable=self.extra_duration_ms,
            width=10,
            state="disabled"
        )

        self.duration_box.pack(
            side="left",
            padx=7
        )

        ttk.Label(
            duration_row,
            text="milisegundos"
        ).pack(side="left")

        ttk.Label(
            behavior,
            text="Se suma a la duración original de cada nota.",
            foreground="#666"
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        ttk.Separator(behavior).pack(
            fill="x",
            pady=5
        )

        ttk.Checkbutton(
            behavior,
            text=(
                "Cortar la nota anterior cuando empieza otra "
                "(modo monofónico)"
            ),
            variable=self.monophonic
        ).pack(anchor="w")

        ttk.Label(
            behavior,
            text="BF y el oponente tienen secuencias independientes.",
            foreground="#666"
        ).pack(
            anchor="w",
            pady=(4, 0)
        )

        exports = ttk.LabelFrame(
            content,
            text="EXPORT FILE NAMES",
            padding=10
        )

        exports.pack(
            fill="x",
            pady=8
        )

        self.entry_row(
            exports,
            "BF Voices",
            self.bf_export_name
        )

        self.entry_row(
            exports,
            "Opponent Voices",
            self.opponent_export_name
        )

        self.entry_row(
            exports,
            "Preview Mix",
            self.mix_export_name
        )

        ttk.Label(
            exports,
            text="La extensión .ogg se agrega automáticamente.",
            foreground="#666"
        ).pack(
            anchor="w",
            pady=(4, 0)
        )

        mixer = ttk.LabelFrame(
            content,
            text="PREVIEW MIXER",
            padding=12
        )

        mixer.pack(
            fill="x",
            pady=8
        )

        ttk.Label(
            mixer,
            text="Elegí las pistas que querés incluir:"
        ).pack(anchor="w")

        checks = ttk.Frame(mixer)
        checks.pack(
            fill="x",
            pady=9
        )

        ttk.Checkbutton(
            checks,
            text="Instrumental",
            variable=self.instrumental_enabled,
            command=self.update_requirements
        ).pack(
            side="left",
            padx=(0, 18)
        )

        ttk.Checkbutton(
            checks,
            text="BF Voices",
            variable=self.bf_enabled,
            command=self.update_requirements
        ).pack(
            side="left",
            padx=(0, 18)
        )

        ttk.Checkbutton(
            checks,
            text="Opponent Voices",
            variable=self.opponent_enabled,
            command=self.update_requirements
        ).pack(side="left")

        ttk.Label(
            mixer,
            text="Estado de los requisitos:",
            font=("Segoe UI", 9, "bold")
        ).pack(
            anchor="w",
            pady=(7, 2)
        )

        ttk.Label(
            mixer,
            textvariable=self.requirements,
            wraplength=850,
            justify="left"
        ).pack(
            anchor="w",
            fill="x"
        )

        buttons = ttk.Frame(mixer)
        buttons.pack(
            fill="x",
            pady=(12, 2)
        )

        self.play_button = ttk.Button(
            buttons,
            text="▶ PLAY PREVIEW",
            command=self.play_preview,
            state="disabled"
        )

        self.play_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 4)
        )

        self.stop_button = ttk.Button(
            buttons,
            text="■ STOP",
            command=self.stop_preview,
            state="disabled"
        )

        self.stop_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=4
        )

        self.export_button = ttk.Button(
            buttons,
            text="EXPORT SELECTED",
            command=self.export_selected,
            state="disabled"
        )

        self.export_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(4, 0)
        )

        ttk.Separator(content).pack(
            fill="x",
            pady=12
        )

        ttk.Label(
            content,
            textvariable=self.status,
            wraplength=850,
            foreground="#235c9b"
        ).pack(
            anchor="w",
            pady=(2, 5)
        )

    def file_row(self, parent, label, variable, command):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)

        ttk.Label(
            row,
            text=label,
            width=25
        ).pack(side="left")

        ttk.Entry(
            row,
            textvariable=variable
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=7
        )

        ttk.Button(
            row,
            text="Browse...",
            command=command
        ).pack(side="right")

    def entry_row(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)

        ttk.Label(
            row,
            text=label,
            width=25
        ).pack(side="left")

        ttk.Entry(
            row,
            textvariable=variable
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=7
        )

    def volume_row(self, parent, label, variable, display):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=5)

        ttk.Label(
            row,
            text=label,
            width=25
        ).pack(side="left")

        ttk.Spinbox(
            row,
            from_=0,
            to=58,
            increment=1,
            textvariable=variable,
            width=8
        ).pack(
            side="left",
            padx=7
        )

        ttk.Label(
            row,
            textvariable=display,
            width=8
        ).pack(side="left")

    def update_note_controls(self):
        state = (
            "normal"
            if self.extend_notes.get()
            else "disabled"
        )

        self.duration_box.configure(
            state=state
        )

        self.update_requirements()

    def connect_validation(self):
        variables = [
            self.chart,
            self.instrumental,
            self.bf_sf2,
            self.opponent_sf2,
            self.output,
            self.difficulty,
            self.bf_gain,
            self.opponent_gain,
            self.sync_percent,
            self.extra_duration_ms,
            self.bf_export_name,
            self.opponent_export_name,
            self.mix_export_name
        ]

        for variable in variables:
            variable.trace_add(
                "write",
                lambda *_: self.update_requirements()
            )

        self.bf_gain.trace_add(
            "write",
            lambda *_: self.update_gain_labels()
        )

        self.opponent_gain.trace_add(
            "write",
            lambda *_: self.update_gain_labels()
        )

    def update_gain_labels(self):
        try:
            self.bf_gain_label.set(
                f"+{int(self.bf_gain.get())} dB"
            )
        except (ValueError, tk.TclError):
            self.bf_gain_label.set("Inválido")

        try:
            self.opponent_gain_label.set(
                f"+{int(self.opponent_gain.get())} dB"
            )
        except (ValueError, tk.TclError):
            self.opponent_gain_label.set("Inválido")

    def update_sync_label(self, value):
        self.sync_label.set(
            f"{int(float(value))}%"
        )

        self.update_requirements()

    def valid_file(self, variable):
        value = variable.get().strip()

        return bool(value) and Path(value).is_file()

    def missing_requirements(self):
        selected_inst = self.instrumental_enabled.get()
        selected_bf = self.bf_enabled.get()
        selected_opponent = self.opponent_enabled.get()

        missing = []

        if not any([
            selected_inst,
            selected_bf,
            selected_opponent
        ]):
            missing.append(
                "Seleccioná al menos una pista del mixer."
            )

        voices_selected = selected_bf or selected_opponent

        try:
            sync = int(self.sync_percent.get())
        except (ValueError, tk.TclError):
            sync = 70
            missing.append(
                "El porcentaje de sincronización no es válido."
            )

        needs_analysis = (
            voices_selected and sync > 0
        )

        if selected_inst and not self.valid_file(self.instrumental):
            missing.append(
                "Instrumental: necesaria para reproducirla."
            )

        if needs_analysis and not self.valid_file(self.instrumental):
            missing.append(
                "Instrumental: necesaria para detectar las notas coincidentes."
            )

        if not find_tool("ffmpeg") and not find_tool("ffmpeg.exe"):
            missing.append(
                "FFmpeg: necesario para procesar el audio."
            )

        if voices_selected:
            if not self.valid_file(self.chart):
                missing.append(
                    "Chart JSON: necesario para generar las voces."
                )

            if not self.difficulty.get().strip():
                missing.append(
                    "Dificultad: seleccioná una dificultad."
                )

            if not find_tool("fluidsynth") and not find_tool("fluidsynth.exe"):
                missing.append(
                    "FluidSynth: necesario para sintetizar las voces."
                )

            if importlib.util.find_spec("mido") is None:
                missing.append(
                    "Mido: instalalo con py -m pip install mido."
                )

            if needs_analysis and importlib.util.find_spec("numpy") is None:
                missing.append(
                    "NumPy: instalalo con py -m pip install numpy."
                )

        if selected_bf and not self.valid_file(self.bf_sf2):
            missing.append(
                "Boyfriend SF2: necesario para generar BF Voices."
            )

        if selected_opponent and not self.valid_file(self.opponent_sf2):
            missing.append(
                "Opponent SF2: necesario para generar Opponent Voices."
            )

        return missing

    def get_export_names(self):
        bf_name = safe_ogg_filename(
            self.bf_export_name.get(),
            "Voices-bf.ogg"
        )

        opponent_name = safe_ogg_filename(
            self.opponent_export_name.get(),
            "Voices-opponent.ogg"
        )

        mix_name = safe_ogg_filename(
            self.mix_export_name.get(),
            "Preview-Mix.ogg"
        )

        names = [
            bf_name.lower(),
            opponent_name.lower(),
            mix_name.lower()
        ]

        if len(set(names)) != 3:
            raise ValueError(
                "Los tres nombres de exportación deben ser diferentes."
            )

        return bf_name, opponent_name, mix_name

    def update_requirements(self):
        missing = self.missing_requirements()

        lines = [
            "• " + item
            for item in missing
        ]

        ffplay = (
            find_tool("ffplay")
            or find_tool("ffplay.exe")
        )

        if not ffplay:
            lines.append(
                "• FFplay: necesario para PLAY PREVIEW, no para exportar."
            )

        if not lines:
            self.requirements.set(
                "Todo listo. Podés reproducir la mezcla o exportarla."
            )
        else:
            self.requirements.set(
                "\n".join(lines)
            )

        if self.busy:
            self.play_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
            return

        preview_ready = (
            not missing and bool(ffplay)
        )

        self.play_button.configure(
            state="normal" if preview_ready else "disabled"
        )

        try:
            self.get_export_names()
            names_valid = True
        except ValueError:
            names_valid = False

        export_ready = (
            not missing
            and names_valid
            and bool(self.output.get().strip())
        )

        self.export_button.configure(
            state="normal" if export_ready else "disabled"
        )

    def get_config(self):
        missing = self.missing_requirements()

        if missing:
            raise ValueError(
                "\n".join(missing)
            )

        bf_gain = int(self.bf_gain.get())
        opponent_gain = int(self.opponent_gain.get())

        if not 0 <= bf_gain <= 58:
            raise ValueError(
                "El volumen de BF debe estar entre 0 y 58 dB."
            )

        if not 0 <= opponent_gain <= 58:
            raise ValueError(
                "El volumen del oponente debe estar entre 0 y 58 dB."
            )

        try:
            extra_duration = int(
                self.extra_duration_ms.get()
            )
        except (ValueError, tk.TclError):
            raise ValueError(
                "La duración adicional no es válida."
            )

        if not 0 <= extra_duration <= 1000:
            raise ValueError(
                "La duración adicional debe estar entre 0 y 1000 ms."
            )

        return {
            "instrumental_enabled": self.instrumental_enabled.get(),
            "bf_enabled": self.bf_enabled.get(),
            "opponent_enabled": self.opponent_enabled.get(),

            "instrumental": (
                Path(self.instrumental.get())
                if self.valid_file(self.instrumental)
                else None
            ),

            "chart": (
                Path(self.chart.get())
                if self.valid_file(self.chart)
                else None
            ),

            "bf_sf2": (
                Path(self.bf_sf2.get())
                if self.valid_file(self.bf_sf2)
                else None
            ),

            "opponent_sf2": (
                Path(self.opponent_sf2.get())
                if self.valid_file(self.opponent_sf2)
                else None
            ),

            "difficulty": self.difficulty.get(),

            "sync_percent": int(self.sync_percent.get()),

            "bf_gain": bf_gain,
            "opponent_gain": opponent_gain,

            "extra_duration_ms": (
                extra_duration
                if self.extend_notes.get()
                else 0
            ),

            "monophonic": self.monophonic.get()
        }

    def set_busy(self, value):
        self.busy = value

        if value:
            self.play_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
        else:
            self.update_requirements()

    def pick_chart(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.chart.set(path)

        try:
            data = load_json(path)
            difficulties = get_vslice_difficulties(data)

            if difficulties:
                self.difficulty_box.configure(
                    values=difficulties
                )

                self.difficulty.set(
                    "hard" if "hard" in difficulties
                    else difficulties[0]
                )

            else:
                self.difficulty_box.configure(
                    values=["Psych / Legacy"]
                )

                self.difficulty.set("Psych / Legacy")

            read_chart(
                Path(path),
                self.difficulty.get()
            )

            self.status.set(
                "Chart cargado correctamente."
            )

        except Exception as error:
            self.difficulty_box.configure(
                values=[]
            )

            self.difficulty.set("")

            self.status.set(
                "No se pudo cargar el chart."
            )

            messagebox.showerror(
                "Chart inválido",
                str(error)
            )

        self.update_requirements()

    def pick_instrumental(self):
        path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Audio",
                    "*.ogg *.wav *.mp3 *.flac *.m4a"
                ),
                ("All files", "*.*")
            ]
        )

        if path:
            self.instrumental.set(path)

    def pick_metadata(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Metadata JSON", "*.json"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.metadata.set(path)

            opponent_id = get_opponent_character(path)

            if self.opponent_export_name.get() in (
                "",
                "Voices-opponent.ogg"
            ):
                self.opponent_export_name.set(
                    f"Voices-{opponent_id}.ogg"
                )

    def pick_bf(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("SoundFont 2", "*.sf2"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.bf_sf2.set(path)

    def pick_opponent(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("SoundFont 2", "*.sf2"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.opponent_sf2.set(path)

    def pick_output(self):
        path = filedialog.askdirectory()

        if path:
            self.output.set(path)

    def play_preview(self):
        try:
            config = self.get_config()
        except Exception as error:
            messagebox.showerror(
                "Faltan requisitos",
                str(error)
            )
            return

        ffmpeg = (
            find_tool("ffmpeg")
            or find_tool("ffmpeg.exe")
        )

        fluidsynth = (
            find_tool("fluidsynth")
            or find_tool("fluidsynth.exe")
        )

        ffplay = (
            find_tool("ffplay")
            or find_tool("ffplay.exe")
        )

        if not ffmpeg or not ffplay:
            messagebox.showerror(
                "Dependencias",
                "FFmpeg y FFplay son necesarios para reproducir el preview."
            )
            return

        if (
            config["bf_enabled"]
            or config["opponent_enabled"]
        ) and not fluidsynth:
            messagebox.showerror(
                "Dependencias",
                "FluidSynth es necesario para generar las voces."
            )
            return

        self.set_busy(True)

        self.stop_button.configure(
            state="normal"
        )

        self.status.set(
            "Generando preview temporal..."
        )

        self.preview_stop_event = threading.Event()

        stop_event = self.preview_stop_event

        self.preview_thread = threading.Thread(
            target=self.preview_worker,
            args=(
                config,
                ffmpeg,
                fluidsynth,
                ffplay,
                stop_event
            ),
            daemon=True
        )

        self.preview_thread.start()

    def preview_worker(
        self,
        config,
        ffmpeg,
        fluidsynth,
        ffplay,
        stop_event
    ):
        error_message = None
        stats = None

        try:
            with tempfile.TemporaryDirectory(
                prefix="fnf_preview_"
            ) as temp_folder:

                temp = Path(temp_folder)

                sources, voices, detected, fallback = build_selected_audio(
                    config,
                    ffmpeg,
                    fluidsynth,
                    temp
                )

                if stop_event.is_set():
                    return

                preview_path = temp / "Preview-Temporary.ogg"

                mix_audio_files(
                    ffmpeg,
                    sources,
                    preview_path
                )

                if stop_event.is_set():
                    return

                flags = getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                )

                process = subprocess.Popen(
                    [
                        ffplay,
                        "-nodisp",
                        "-autoexit",
                        "-loglevel", "error",
                        str(preview_path)
                    ],
                    creationflags=flags
                )

                with self.preview_lock:
                    if stop_event.is_set():
                        process.terminate()
                    else:
                        self.preview_process = process

                stats = (
                    len(voices),
                    detected,
                    fallback
                )

                self.root.after(
                    0,
                    lambda: self.status.set(
                        "Reproduciendo preview..."
                    )
                )

                while process.poll() is None:
                    if stop_event.wait(0.1):
                        try:
                            process.terminate()
                        except OSError:
                            pass

                        break

                process.wait()

        except Exception as error:
            error_message = str(error)

        finally:
            with self.preview_lock:
                self.preview_process = None

            self.root.after(
                0,
                lambda: self.preview_finished(
                    error_message,
                    stop_event.is_set(),
                    stats
                )
            )

    def preview_finished(
        self,
        error_message,
        stopped,
        stats
    ):
        self.set_busy(False)

        self.stop_button.configure(
            state="disabled"
        )

        if error_message:
            self.status.set(
                "Error preparando el preview."
            )

            messagebox.showerror(
                "Preview error",
                error_message
            )

        elif stopped:
            self.status.set(
                "Preview detenido. Archivos temporales eliminados."
            )

        else:
            self.status.set(
                "Preview terminado."
            )

            if stats:
                voice_count, detected, fallback = stats

                self.status.set(
                    f"Preview terminado. "
                    f"Voces: {voice_count}. "
                    f"Tonos detectados: {detected}. "
                    f"Sin detección: {fallback}."
                )

    def stop_preview(self):
        if self.preview_stop_event:
            self.preview_stop_event.set()

        with self.preview_lock:
            process = self.preview_process

            if process is not None:
                try:
                    process.terminate()
                except OSError:
                    pass

        if (
            self.preview_thread
            and self.preview_thread.is_alive()
        ):
            self.status.set(
                "Deteniendo preview..."
            )

    def export_selected(self):
        try:
            config = self.get_config()

            bf_name, opponent_name, mix_name = self.get_export_names()

            if len({
                bf_name.lower(),
                opponent_name.lower(),
                mix_name.lower()
            }) != 3:
                raise ValueError(
                    "Los tres nombres de exportación deben ser diferentes."
                )

        except Exception as error:
            messagebox.showerror(
                "Configuración inválida",
                str(error)
            )
            return

        ffmpeg = (
            find_tool("ffmpeg")
            or find_tool("ffmpeg.exe")
        )

        fluidsynth = (
            find_tool("fluidsynth")
            or find_tool("fluidsynth.exe")
        )

        if not ffmpeg:
            messagebox.showerror(
                "Dependencias",
                "FFmpeg es necesario para exportar."
            )
            return

        if (
            config["bf_enabled"]
            or config["opponent_enabled"]
        ) and not fluidsynth:
            messagebox.showerror(
                "Dependencias",
                "FluidSynth es necesario para generar las voces."
            )
            return

        config["output"] = Path(self.output.get())

        config["bf_filename"] = bf_name
        config["opponent_filename"] = opponent_name
        config["mix_filename"] = mix_name

        self.set_busy(True)

        self.status.set(
            "Generando archivos de exportación..."
        )

        threading.Thread(
            target=self.export_worker,
            args=(
                config,
                ffmpeg,
                fluidsynth
            ),
            daemon=True
        ).start()

    def export_worker(
        self,
        config,
        ffmpeg,
        fluidsynth
    ):
        try:
            output = config["output"]

            output.mkdir(
                parents=True,
                exist_ok=True
            )

            exported = []

            with tempfile.TemporaryDirectory(
                prefix="fnf_export_"
            ) as temp_folder:

                temp = Path(temp_folder)

                sources, voices, detected, fallback = build_selected_audio(
                    config,
                    ffmpeg,
                    fluidsynth,
                    temp
                )

                if "bf" in voices:
                    destination = (
                        output / config["bf_filename"]
                    )

                    shutil.copy2(
                        voices["bf"],
                        destination
                    )

                    exported.append(destination.name)

                if "opponent" in voices:
                    destination = (
                        output / config["opponent_filename"]
                    )

                    shutil.copy2(
                        voices["opponent"],
                        destination
                    )

                    exported.append(destination.name)

                mix_path = (
                    output / config["mix_filename"]
                )

                mix_audio_files(
                    ffmpeg,
                    sources,
                    mix_path
                )

                exported.append(
                    mix_path.name
                )

            self.root.after(
                0,
                lambda: self.export_finished(
                    output,
                    exported,
                    detected,
                    fallback
                )
            )

        except Exception as error:
            message = str(error)

            self.root.after(
                0,
                lambda: self.export_failed(message)
            )

    def export_finished(
        self,
        output,
        exported,
        detected,
        fallback
    ):
        self.set_busy(False)

        details = "\n".join(exported)

        details += (
            f"\n\nTonos detectados: {detected}"
            f"\nSin detección: {fallback}"
        )

        self.status.set(
            "Exportación terminada."
        )

        messagebox.showinfo(
            "Exportación terminada",
            f"Archivos guardados en:\n{output}\n\n{details}"
        )

    def export_failed(self, message):
        self.set_busy(False)

        self.status.set(
            "No se pudo exportar."
        )

        messagebox.showerror(
            "Export error",
            message
        )

    def close(self):
        self.stop_preview()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    App(root)

    root.mainloop()
