import io
import os
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta

from lark import Lark, Transformer, Tree


def parse_time(s):
    return datetime.strptime(s, "%H:%M:%S")


def take_line(text):
    buf = io.StringIO(text)
    return buf.readline().strip()


@dataclass
class Stage:
    end: datetime
    name: str
    metrics: Dict[str, float]


@dataclass
class EpochLog:
    epoch: int
    epochs: int
    stages: List[Stage]


grammar = r"""
    text : epoch_log* metric_line?
    epoch_log : epoch_line metric_line? metric_line?

    epoch_line : "Epoch" int "/" int
    metric_line : time sname "-" metrics

    sname : /train|valid/
    metrics: [metric ("," metric)*]
    metric : mname ":" mvalue

    time : /[0-9]{2}:[0-9]{2}:[0-9]{2}/
    mname : /[a-z|A-Z|0-9]+/
    mvalue : DECIMAL
    int : INT

    START_LINE : /[0-9]{2}:[0-9]{2}:[0-9]{2} Start training/
    SAVE_LINE : /Save learner to (\/[a-zA-Z0-9-]+)+/
    ERROR_MSG : /Traceback.*?ignored\./s
    TF_MSG : /[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}: W .*?Socket closed/s
    TF_WARN : /WARNING\:.*/
    TF_UPDATE : /Instructions.*?instead\./s
    COCO_LOAD : /Loading and preparing.*?DONE.*?DONE.*?DONE \(t=[0-9]{1,3}\.[0-9]{1,3}s\)\./s
    COCO_RES : /Average (Precision|Recall).*?= 0\.[0-9]{3}/s

    %ignore START_LINE
    %ignore SAVE_LINE
    %ignore ERROR_MSG
    %ignore TF_MSG
    %ignore TF_WARN
    %ignore TF_UPDATE
    %ignore COCO_LOAD
    %ignore COCO_RES


    %import common.LETTER
    %import common.INT
    %import common.DIGIT
    %import common.DECIMAL
    %import common.WS
    %ignore WS
    """


def filter_log(text):
    parser = Lark(grammar, start='text', parser='lalr')
    lines = [take_line(text)]
    tree = parser.parse(text)
    Tree.val = property(lambda t: t.children[0].value)
    for e in tree.children:
        for l in e.children:
            if l.data == 'epoch_line':
                lines.append("Epoch %s/%s" % (l.children[0].val, l.children[1].val))
            elif l.data == 'metric_line':
                end = l.children[0].val
                stage = l.children[1].val
                metrics = ", ".join([
                    f"{m.children[0].val}: {m.children[1].val}"
                    for m in l.children[2].children
                ])
                lines.append("%s %s - %s" % (end, stage, metrics))
    delattr(Tree, "val")
    return os.linesep.join(lines)


class TreeToLog(Transformer):
    def time(self, s):
        return parse_time(s[0])

    def sname(self, s):
        return s[0].value

    def mname(self, s):
        return s[0].value

    def mvalue(self, n):
        return float(n[0])

    def int(self, n):
        return int(n[0])

    metric = tuple
    metrics = dict

    def metric_line(self, s):
        return Stage(*s)

    def epoch_line(self, s):
        return s

    def epoch_log(self, s):
        return EpochLog(s[0][0], s[0][1], s[1:])

    text = list


def parse_train_start(text):
    first_line = take_line(text)
    assert first_line.endswith("Start training")
    return parse_time(first_line[:8])


def remove_duplicate_epochs(res: List[EpochLog]):
    new_res = [res[0]]
    for l in res[1:]:
        if l.epoch == new_res[-1].epoch:
            new_res.pop(-1)
        new_res.append(l)
    return new_res


def fix_datetime(res: List[EpochLog], train_start: datetime):
    date_now = train_start.date()
    time_now = train_start.time()
    for l in res:
        for s in l.stages:
            dt = s.end
            if dt.time() < time_now:
                date_now = date_now + timedelta(days=1)
            time_now = dt.time()
            s.end = datetime.combine(date_now, time_now)
    return res


def parse_log(text):
    parser = Lark(grammar, start='text', parser='lalr', transformer=TreeToLog())
    train_start = parse_train_start(text)
    res = parser.parse(text)
    if isinstance(res[-1], Stage):
        final_stage = res[-1]
        res = res[:-1]
    else:
        final_stage = None
    res = remove_duplicate_epochs(res)
    res = fix_datetime(res, train_start)
    if final_stage is not None:
        res.append(final_stage)
    return train_start, res


