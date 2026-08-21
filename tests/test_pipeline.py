import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from video_subtitle_pipeline import pipeline


class SegmentTests(unittest.TestCase):
    def test_fixed_segments_clamps_tail(self) -> None:
        segments = pipeline.fixed_segments(12.25, 5)
        self.assertEqual(len(segments), 3)
        self.assertEqual((segments[-1].start, segments[-1].end), (10, 12.25))

    def test_fixed_segments_merges_subsecond_container_tail(self) -> None:
        segments = pipeline.fixed_segments(10.004, 5)
        self.assertEqual(len(segments), 2)
        self.assertEqual((segments[-1].start, segments[-1].end), (5, 10.004))

    def test_time_round_trip(self) -> None:
        self.assertEqual(pipeline.parse_time("00:01:02,345"), 62.345)
        self.assertEqual(pipeline.format_srt_time(62.345), "00:01:02,345")

    def test_short_trailing_noise_is_removed(self) -> None:
        segments = [
            pipeline.Segment("1", 0, 5, text="Complete sentence."),
            pipeline.Segment("2", 5, 5.5, text="P."),
        ]
        self.assertTrue(pipeline.drop_short_trailing_fragment(segments, 3))
        self.assertEqual(segments[-1].text, "")


class OverlapTests(unittest.TestCase):
    def test_audio_windows_add_context_and_clamp_to_video(self) -> None:
        segments = pipeline.fixed_segments(10, 5)
        extracted: list[tuple[float, float]] = []

        def extract(*_: object, start: float, end: float, **__: object) -> None:
            extracted.append((start, end))

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(pipeline, "extract_audio_window", side_effect=extract):
                pipeline.transcribe_missing_segments(
                    Path("video.mp4"),
                    segments,
                    duration=10,
                    workdir=Path(directory),
                    audio_stream=0,
                    overlap=1,
                    workers=1,
                    transcriber=lambda audio: f"raw {audio.stem}",
                )
        self.assertEqual(extracted, [(0, 6), (4, 10)])
        self.assertEqual(segments[0].asr_audio_start, 0)
        self.assertEqual(segments[0].asr_audio_end, 6)
        self.assertEqual(segments[1].raw_asr_text, "raw segment-0002")
        self.assertEqual(segments[1].text, "")

    def test_zero_overlap_promotes_raw_text_directly(self) -> None:
        segment = pipeline.Segment("1", 0, 5)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(pipeline, "extract_audio_window"):
                pipeline.transcribe_missing_segments(
                    Path("video.mp4"),
                    [segment],
                    duration=5,
                    workdir=Path(directory),
                    audio_stream=0,
                    overlap=0,
                    workers=1,
                    transcriber=lambda _: "complete words",
                )
        self.assertEqual(segment.text, "complete words")

    def test_deterministic_stitch_removes_exact_overlap(self) -> None:
        segments = [
            pipeline.Segment(
                "1", 0, 5, raw_asr_text="we calibrate the robot arm now", asr_audio_start=0
            ),
            pipeline.Segment(
                "2", 5, 10, raw_asr_text="robot arm now and save it", asr_audio_start=4
            ),
        ]
        pipeline.stitch_deterministic(segments)
        self.assertEqual(segments[0].text, "we calibrate the robot arm now")
        self.assertEqual(segments[1].text, "and save it")

    def test_llm_stitch_uses_neighbor_context_but_only_updates_targets(self) -> None:
        segments = [
            pipeline.Segment(
                str(index),
                index * 5,
                (index + 1) * 5,
                raw_asr_text=f"raw {index}",
                asr_audio_start=max(0, index * 5 - 1),
            )
            for index in range(3)
        ]
        calls: list[str] = []

        def fake(prompt: str, **kwargs: object) -> dict[str, str]:
            calls.append(prompt)
            validator = kwargs["validator"]
            schema = kwargs["schema"]
            ids = schema["properties"]["segments"]["items"]["properties"]["id"]["enum"]
            payload = {"segments": [{"id": identifier, "text": f"stitched {identifier}"} for identifier in ids]}
            return validator(payload)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(pipeline, "run_llm_json", side_effect=fake):
                pipeline.stitch_with_llm(
                    segments,
                    provider="codex",
                    model="test",
                    effort="low",
                    command_template=None,
                    batch_size=1,
                    workdir=Path(directory),
                )
        self.assertEqual([segment.text for segment in segments], ["stitched 0", "stitched 1", "stitched 2"])
        self.assertIn('"output_required":false', calls[1])

    def test_stitch_does_not_replace_manual_anchor_text(self) -> None:
        segments = [
            pipeline.Segment("manual", 0, 5, text="Reviewed source text."),
            pipeline.Segment(
                "asr",
                5,
                10,
                raw_asr_text="Reviewed source text continues here.",
                asr_audio_start=4,
            ),
        ]
        pipeline.stitch_deterministic(segments)
        self.assertEqual(segments[0].text, "Reviewed source text.")


class TimestampedASRTests(unittest.TestCase):
    def test_parse_verbose_word_timestamps(self) -> None:
        result = pipeline.parse_asr_result(
            {
                "text": "Move the robot arm.",
                "duration": 6.0,
                "words": [
                    {"word": "Move", "start": 0.8, "end": 1.1},
                    {"word": "the", "start": 1.1, "end": 1.3},
                    {"word": "robot", "start": 4.8, "end": 5.2},
                    {"word": "arm.", "start": 5.2, "end": 5.6},
                ],
            }
        )
        self.assertEqual(result.duration, 6.0)
        self.assertEqual(result.words[2].word, "robot")

    def test_parse_rejects_text_only_response(self) -> None:
        with self.assertRaisesRegex(pipeline.PipelineError, "did not return word timestamps"):
            pipeline.parse_asr_result({"text": "Text only."})

    def test_timestamp_midpoints_assign_words_to_nominal_anchors(self) -> None:
        segments = pipeline.fixed_segments(10, 5)
        result = pipeline.ASRResult(
            text="Move the robot arm.",
            duration=10,
            words=[
                pipeline.ASRWord("Move", 0.8, 1.1),
                pipeline.ASRWord("the", 1.1, 1.3),
                pipeline.ASRWord("robot", 4.8, 5.2),
                pipeline.ASRWord("arm.", 5.2, 5.6),
            ],
        )
        pipeline.assign_timestamped_words(segments, result, duration=10)
        self.assertEqual(segments[0].text, "Move the")
        self.assertEqual(segments[1].text, "robot arm.")
        self.assertEqual(segments[1].asr_words[0].start, 4.8)
        self.assertEqual(segments[0].asr_audio_start, 0)
        self.assertEqual(segments[0].asr_audio_end, 10)

    def test_timestamp_assignment_preserves_reviewed_anchor(self) -> None:
        segments = [
            pipeline.Segment("reviewed", 0, 5, text="Reviewed wording."),
            pipeline.Segment("missing", 5, 10),
        ]
        result = pipeline.ASRResult(
            text="raw first raw second",
            words=[
                pipeline.ASRWord("raw", 1.0, 1.2),
                pipeline.ASRWord("second", 6.0, 6.5),
            ],
        )
        pipeline.assign_timestamped_words(segments, result, duration=10)
        self.assertEqual(segments[0].text, "Reviewed wording.")
        self.assertEqual(segments[1].text, "second")


class ConfigTests(unittest.TestCase):
    def test_config_loads_defaults_and_cli_overrides_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(
                json.dumps({"audio_overlap": 1.5, "asr_workers": 2, "burn": True}),
                encoding="utf-8",
            )
            args = pipeline.parse_args(
                ["--config", str(config), "--asr-workers", "7", "video.mp4"]
            )
        self.assertEqual(args.audio_overlap, 1.5)
        self.assertEqual(args.asr_workers, 7)
        self.assertEqual(args.asr_mode, "whole")
        self.assertTrue(args.burn)

    def test_unknown_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text('{"asr_wokers": 4}', encoding="utf-8")
            with self.assertRaisesRegex(pipeline.PipelineError, "Unknown config"):
                pipeline.parse_args(["--config", str(config), "video.mp4"])


class SubtitleTests(unittest.TestCase):
    def test_bilingual_cue_keeps_source_above_translation(self) -> None:
        segment = pipeline.Segment(
            "1",
            0,
            5,
            text="Move the robot arm.",
            translation_parts=["로봇 팔을 이동합니다."],
        )
        cues = pipeline.build_cues(
            [segment], use_source_text=False, include_source=True, max_parts=1
        )
        self.assertEqual(cues[0].text, "Move the robot arm.\n로봇 팔을 이동합니다.")


class PrivacyTests(unittest.TestCase):
    def test_committed_example_has_placeholder_endpoint(self) -> None:
        example = Path(__file__).parents[1] / "config.example.json"
        payload = json.loads(example.read_text(encoding="utf-8"))
        self.assertIn("your-asr-host.example", payload["asr_url"])


if __name__ == "__main__":
    unittest.main()
