import video_color

# video_color.video_predict("videos/hepburn.mp4", "images/examples/01.jpg", "output/color.mp4")
# video_color.video_predict("videos/0001.mp4", "images/0001/*.png", "output/0001.mp4")

video_color.image_predict("images/blackswan/*.png", "images/examples/00000*.png", "output/blackswan")
