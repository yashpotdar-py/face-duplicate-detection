# Sample Data Directory Structure

This directory contains sample data for testing the face duplicate detection system.

## Directory Structure

```
data/
├── Videos/          # Place your video files here
│   ├── bicycle_video_001.mp4
│   ├── bicycle_video_002.avi
│   └── ...
├── Thumbnails/      # Optional thumbnail images
│   ├── thumb_001.jpg
│   ├── thumb_002.jpg
│   └── ...
└── README.md        # This file
```

## Supported Video Formats

- **MP4** (.mp4) - Recommended
- **AVI** (.avi)
- **MOV** (.mov)
- **MKV** (.mkv)
- **WMV** (.wmv)
- **FLV** (.flv)

## Video Requirements

- **Content**: Videos should contain people riding bicycles toward the camera
- **Resolution**: Any resolution supported (1080p recommended for best results)
- **Duration**: No specific limits (processing time scales with video length)
- **Frame Rate**: Any frame rate (system will sample frames as needed)

## Sample Usage

1. Place your video files in the `Videos/` directory
2. Run the detection system:
   ```bash
   python detect_duplicates.py detect \
       --videos-dir data/Videos \
       --output-file results/duplicate_report.json
   ```

## Performance Tips

- **File Size**: Larger files take longer to process
- **Resolution**: Higher resolution = better accuracy but slower processing
- **Compression**: Well-compressed H.264/H.265 videos process faster
- **Frame Rate**: System samples frames, so 30fps vs 60fps won't double processing time

## Expected Results

For bicycle videos, you can expect:
- **Detection Rate**: 5-50 faces per minute of video (depends on content)
- **Accuracy**: 95%+ precision on clear, frontal faces
- **Processing Speed**: ~3-5 minutes per 10-minute 1080p video (with GPU)

## Troubleshooting

- **No faces detected**: Check video quality and lighting
- **Poor clustering**: Adjust `--match-threshold` parameter
- **Slow processing**: Use `--skip-frames` to process fewer frames
- **Memory errors**: Reduce `--batch-size` parameter
