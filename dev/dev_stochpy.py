#!/usr/bin/env python3

# Purpose: make sure birth_death.psc is found

from src.synthetic import BirthDeathProcess

b = BirthDeathProcess(birthrate=1, deathrate=1, time_final=10)

breakpoint()
