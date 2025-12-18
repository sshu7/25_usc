import random

CHOICES = ["rock", "paper", "scissors"]

WIN_MAP = {
        ("rock", "scissors"): True,
        ("scissors", "paper"): True,
        ("paper", "rock"): True,
        }


def cpu_choice():
    return random.choice(CHOICES)


def decide(user: str, cpu: str) -> str:
    if user == cpu:
        return "draw"
    return "win" if WIN_MAP.get((user, cpu), False) else "lose"
