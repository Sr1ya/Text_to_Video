# Create a working directory
cd /usr/local/bin

# Download latest static FFmpeg build (Linux 64-bit)
sudo curl -L -o ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Extract it
sudo tar -xvf ffmpeg.tar.xz

# Move into the extracted folder
cd ffmpeg-*-amd64-static

# Copy the binaries to /usr/local/bin
sudo cp ffmpeg ffprobe /usr/local/bin/

# Confirm installation
ffmpeg -version
