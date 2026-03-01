with open("C:/Users/kubam/AntigravityProjects/video_automator/video_engine.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(191, 199):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

for i in range(200, 393):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

with open("C:/Users/kubam/AntigravityProjects/video_automator/video_engine.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
