from hhutil.io import read_lines, write_lines

lines = read_lines("/Users/hrvvi/Code/Library/experiments/ImageNet/log/98-2.log")
new_lines = [lines[0]]
is_error = False
for l in lines[1:]:
    if not is_error and (l.startswith("Traceback") or "Unable to destroy remote tensor handles" in l):
        is_error = True
    elif is_error and l.endswith("Start training"):
        is_error = False
    if not is_error and not l.endswith("Start training") and not l.startswith("Save"):
        new_lines.append(l)
write_lines(new_lines, "/Users/hrvvi/Downloads/1.log")