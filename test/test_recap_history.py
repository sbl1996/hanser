from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

i = 561
now = parse("05:49:56")
t_train = 40
t_eval = 1
# train_losses = []
# train_accs = []
# eval_losses = []
# eval_accs = []

res = ""

while i < 600:
    i += 1
    now = now + relativedelta(seconds=t_train)
    train_end = now
    now = now + relativedelta(seconds=t_eval)
    eval_end = now
    offset = i - 562
    s = f"""
Epoch {i}/600
{train_end.strftime("%X")} train - loss: {train_losses[offset]:.4f}, acc: {train_accs[offset]:.4f}
{eval_end.strftime("%X")} valid - loss: {eval_losses[offset]:.4f}, acc: {eval_accs[offset]:.4f}"""
    res += s