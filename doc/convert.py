from moviepy.editor import *

clip = (VideoFileClip("doc\web-teaser2.mp4"))
clip.write_gif("doc\web-teaser2.gif",fps=10)