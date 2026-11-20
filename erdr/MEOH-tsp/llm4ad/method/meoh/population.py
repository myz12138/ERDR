from __future__ import annotations

import math
from threading import Condition, Lock
from typing import List
import numpy as np
import traceback

from ...base import *
from codebleu.syntax_match import calc_syntax_match
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None,
                 defer_generation_finalization: bool = False):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._generation_condition = Condition(self._lock)
        self._generation_finalizing = False
        self._defer_generation_finalization = bool(defer_generation_finalization)
        self._next_gen_pop = []
        self._elitist = []
        self._generation = generation
        self.selection_score_attribute = 'score'

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def elitist(self):
        return self._elitist

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        if self._defer_generation_finalization:
            return self._register_deferred_function(func)
        # we only accept valid functions
        if func.score is None:
            return
        try:
            self._lock.acquire()
            # register to next_gen
            if not self.has_duplicate_function(func):
                self._next_gen_pop.append(func)

            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size or (len(self._next_gen_pop) >= self._pop_size // 4 and self._generation == 0):
                pop = self._population + self._next_gen_pop

                pop_elitist = pop + self._elitist
                # objs = [ind.score for ind in pop_elitist]
                # objs_array = -np.array(objs)
                # nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
                # self._elitist = []
                # for idx in nondom_idx.tolist():
                #     self._elitist.append(pop_elitist[idx])
                # Filter based on scores/objectives
                unique_pop = []
                seen_scores = set()

                for ind in pop_elitist:
                    # Convert score list/array to tuple so it can be hashed
                    score_tuple = tuple(ind.score) 
                    if score_tuple not in seen_scores:
                        unique_pop.append(ind)
                        seen_scores.add(score_tuple)

                pop_elitist = unique_pop

                objs = [ind.score for ind in pop_elitist]
                objs_array = -np.array(objs)
                nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)

                self._elitist = []
                for idx in nondom_idx.tolist():
                    self._elitist.append(pop_elitist[idx])

                crt_pop_size = len(pop)
                dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
                for i in range(crt_pop_size):
                    for j in range(i + 1, crt_pop_size):
                        if (np.array(pop[i].score) >= np.array(pop[j].score)).all():
                            dominated_counts[i, j] = -calc_syntax_match([pop[i].entire_code], pop[j].entire_code, 'python')
                        elif (np.array(pop[j].score) >= np.array(pop[i].score)).all():
                            dominated_counts[j, i] = -calc_syntax_match([pop[j].entire_code], pop[i].entire_code, 'python')
                dominated_counts_ = dominated_counts.sum(0)
                self._population = [pop[i] for i in np.argsort(-dominated_counts_)[:self._pop_size]]  # minus for descending, //5 for keep the original pop_size
                self._next_gen_pop = []
                self._generation += 1

        except Exception as e:
            # print(f"error in registering function to population: {e}")
            traceback.print_exc()
            return
        finally:
            self._lock.release()

    def _register_deferred_function(self, func: Function):
        if func.score is None:
            return None
        with self._generation_condition:
            while self._generation_finalizing:
                self._generation_condition.wait()
            if not self.has_duplicate_function(func):
                self._next_gen_pop.append(func)
            threshold = self._pop_size
            if self._generation == 0:
                threshold = max(1, self._pop_size // 4)
            if len(self._next_gen_pop) < threshold:
                return None
            candidate_pool = self._population + self._next_gen_pop
            self._next_gen_pop = []
            self._generation_finalizing = True
            return candidate_pool

    @staticmethod
    def _valid_score(ind: Function, score_attribute: str) -> bool:
        value = getattr(ind, score_attribute, None)
        return value is not None and not np.isinf(np.asarray(value, dtype=float)).any()

    def _select_survivors(self, candidate_pool: List[Function], score_attribute: str):
        candidates = [
            ind for ind in candidate_pool
            if self._valid_score(ind, score_attribute)
        ]
        if not candidates:
            raise RuntimeError(f"no candidates have a valid {score_attribute}")

        unique_elitist = []
        seen_scores = set()
        for ind in candidates + self._elitist:
            if not self._valid_score(ind, score_attribute):
                continue
            score_tuple = tuple(np.asarray(getattr(ind, score_attribute), dtype=float).ravel().tolist())
            if score_tuple not in seen_scores:
                unique_elitist.append(ind)
                seen_scores.add(score_tuple)

        elitist_scores = -np.asarray(
            [getattr(ind, score_attribute) for ind in unique_elitist], dtype=float
        )
        nondom_idx = NonDominatedSorting().do(
            elitist_scores, only_non_dominated_front=True
        )
        self._elitist = [unique_elitist[idx] for idx in nondom_idx.tolist()]

        size = len(candidates)
        dominated_counts = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                score_i = np.asarray(getattr(candidates[i], score_attribute), dtype=float)
                score_j = np.asarray(getattr(candidates[j], score_attribute), dtype=float)
                if (score_i >= score_j).all():
                    dominated_counts[i, j] = -calc_syntax_match(
                        [candidates[i].entire_code], candidates[j].entire_code, 'python'
                    )
                elif (score_j >= score_i).all():
                    dominated_counts[j, i] = -calc_syntax_match(
                        [candidates[j].entire_code], candidates[i].entire_code, 'python'
                    )
        ranking = np.argsort(-dominated_counts.sum(0))[:self._pop_size]
        self._population = [candidates[idx] for idx in ranking]
        self.selection_score_attribute = score_attribute

    def finalize_deferred_generation(self, candidate_pool: List[Function], score_attribute: str):
        with self._generation_condition:
            if not self._generation_finalizing:
                return
            try:
                self._select_survivors(candidate_pool, score_attribute)
                self._generation += 1
            finally:
                self._generation_finalizing = False
                self._generation_condition.notify_all()

    def has_duplicate_function(self, func: str | Function) -> bool:
        if func.score is None:
            return True

        for i in range(len(self._population)):
            f = self._population[i]
            if str(f) == str(func):
                if func.score[0] > f.score[0]:
                    self._population[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] > f.score[1]:
                    self._population[i] = func
                    return True

        for i in range(len(self._next_gen_pop)):
            f = self._next_gen_pop[i]
            if str(f) == str(func):
                if func.score[0] > f.score[0]:
                    self._next_gen_pop[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] > f.score[1]:
                    self._next_gen_pop[i] = func
                    return True
        return False

    def selection(self) -> Function:
        with self._generation_condition:
            while self._generation_finalizing:
                self._generation_condition.wait()
            score_attribute = self.selection_score_attribute
            population = list(self._population)
        funcs = [
            f for f in population
            if getattr(f, score_attribute, None) is not None
            and not np.isinf(np.array(getattr(f, score_attribute))).any()
        ]

        # AST
        if len(funcs) > 0:
            crt_pop_size = len(funcs)
            dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
            for i in range(crt_pop_size):
                for j in range(i + 1, crt_pop_size):
                    score_i = np.array(getattr(funcs[i], score_attribute))
                    score_j = np.array(getattr(funcs[j], score_attribute))
                    if (score_i >= score_j).all():
                        dominated_counts[i, j] = -calc_syntax_match([funcs[i].entire_code], funcs[j].entire_code, 'python')
                    elif (score_j >= score_i).all():
                        dominated_counts[j, i] = -calc_syntax_match([funcs[j].entire_code], funcs[i].entire_code, 'python')
            dominated_counts_ = dominated_counts.sum(0)
            p = np.exp(dominated_counts_) / np.exp(dominated_counts_).sum()

        return np.random.choice(funcs, p=p, replace=False)
