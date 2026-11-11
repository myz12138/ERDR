import numpy as np
import json
import random
import time
from .eoh_interface_EC import InterfaceEC
# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator
        self.dds_eps_rel = float(max(getattr(paras, "dds_eps_rel", 1e-3), 0.0))
        self.dds_softmax_tau = float(max(getattr(paras, "dds_softmax_tau", 1.0), 1e-6))
        self.dds_novelty_weight = float(max(getattr(paras, "dds_novelty_weight", 0.05), 0.0))
        self.memory_hook = getattr(paras, "memory_hook", None)
        self.memory_min_delta = float(getattr(paras, "memory_min_delta", 0.0))
        self.trainval_enabled = bool(getattr(paras, "trainval_enabled", False))
        self.validation_problem = getattr(paras, "validation_problem", None)
        self.trainval_select_size = int(max(getattr(paras, "trainval_select_size", self.pop_size), 1))
        self.trainval_mode = str(getattr(paras, "trainval_mode", "legacy")).strip().lower()

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    def _population_management(self, population, size):
        try:
            return self.manage.population_management(
                population,
                size,
                eps_rel=self.dds_eps_rel,
                novelty_weight=self.dds_novelty_weight,
            )
        except TypeError:
            return self.manage.population_management(population, size)

    def _numeric_objective(self, indiv):
        if not isinstance(indiv, dict):
            return None
        value = indiv.get("objective")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _numeric_gap(self, indiv):
        if not isinstance(indiv, dict):
            return None
        other = indiv.get("other_inf")
        if not isinstance(other, dict):
            return None
        value = other.get("avg_gap_pct")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _numeric_avg_obj(self, indiv):
        if not isinstance(indiv, dict):
            return None
        other = indiv.get("other_inf")
        if not isinstance(other, dict):
            return None
        value = other.get("avg_obj")
        if value is None:
            value = other.get("avg_objective")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _numeric_time_cost(self, indiv):
        if not isinstance(indiv, dict):
            return None
        other = indiv.get("other_inf")
        if not isinstance(other, dict):
            return None
        value = other.get("avg_tsp_time_sec")
        if value is None:
            value = other.get("time_cost_sec")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _select_old_individual(self, parent_group, population):
        candidates = []
        if isinstance(parent_group, list):
            candidates.extend(parent_group)
        elif isinstance(parent_group, dict):
            candidates.append(parent_group)

        if not candidates:
            candidates = population

        valid = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if item.get("code") is None:
                continue
            metric = self._numeric_objective(item)
            if metric is None:
                continue
            valid.append((metric, item))

        if not valid:
            return None, None
        valid.sort(key=lambda x: x[0])
        return valid[0][1], valid[0][0]

    def _record_memory_improvements(self, parents, offsprings, population, pop_id, op):
        if self.memory_hook is None:
            return

        if not isinstance(offsprings, list):
            return

        for idx, new_ind in enumerate(offsprings):
            if not isinstance(new_ind, dict):
                continue
            if new_ind.get("code") is None:
                continue
            new_metric = self._numeric_objective(new_ind)
            if new_metric is None:
                continue

            parent_group = parents[idx] if isinstance(parents, list) and idx < len(parents) else None
            old_ind, old_metric = self._select_old_individual(parent_group, population)
            if old_ind is None:
                continue

            delta_perf = old_metric - new_metric
            if delta_perf <= self.memory_min_delta:
                continue
            old_gap = self._numeric_gap(old_ind)
            new_gap = self._numeric_gap(new_ind)

            meta = {
                "population_id": int(pop_id),
                "operator": op,
                "offspring_index": int(idx),
                "delta_threshold": self.memory_min_delta,
                "old_gap_pct": old_gap,
                "new_gap_pct": new_gap,
            }
            try:
                self.memory_hook.maybe_record(old_ind, new_ind, meta=meta)
            except Exception as e:
                if self.debug_mode:
                    print(f"memory hook failed: {e}")

    @staticmethod
    def _normalize_code(code):
        return "\n".join(line.rstrip() for line in str(code).strip().splitlines())

    def _evaluate_population_with_problem(self, population, eval_problem):
        out = []
        if eval_problem is None:
            return out
        for ind in population:
            if not isinstance(ind, dict):
                continue
            code = str(ind.get("code", "") or "")
            if not code.strip():
                continue
            row = dict(ind)
            try:
                metric = eval_problem.evaluate(code)
            except Exception:
                metric = None
            try:
                metric_f = float(metric if metric is not None else 1e10)
            except Exception:
                metric_f = 1e10
            row["objective"] = np.round(metric_f, 5)
            try:
                if hasattr(eval_problem, "get_last_eval_info"):
                    info = eval_problem.get_last_eval_info()
                else:
                    info = None
            except Exception:
                info = None
            if isinstance(info, dict):
                row["other_inf"] = dict(info)
            out.append(row)
        return out

    def _validation_select_for_next_train(self, population, generation_id):
        if not self.trainval_enabled:
            return population
        if self.validation_problem is None:
            return population

        pop_eval_val = self._evaluate_population_with_problem(population, self.validation_problem)
        if not pop_eval_val:
            return population

        keep_size = int(min(self.trainval_select_size, len(pop_eval_val)))
        pop_keep_val = self._population_management(pop_eval_val, keep_size)
        if not pop_keep_val:
            return population

        # Re-evaluate selected population on training groups before next evolution step.
        pop_keep_train = self._evaluate_population_with_problem(pop_keep_val, self.prob)
        if not pop_keep_train:
            return population
        pop_keep_train = self._population_management(pop_keep_train, keep_size)
        if not pop_keep_train:
            return population

        log_row = {
            "generation": int(generation_id),
            "validation_population_size": int(len(pop_eval_val)),
            "selected_size": int(len(pop_keep_train)),
            "selected_objectives_on_validation": [float(ind.get("objective", 1e10)) for ind in pop_keep_val],
            "selected_objectives_on_train": [float(ind.get("objective", 1e10)) for ind in pop_keep_train],
        }
        filename = (
            self.output_path
            + "/results/history/validation_selection_generation_"
            + str(generation_id)
            + ".json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(log_row, f, indent=5)
        except Exception:
            pass
        print(
            f"validation-select@gen{generation_id}: "
            f"{len(pop_eval_val)} -> {len(pop_keep_train)}",
            flush=True,
        )
        return pop_keep_train

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba,
                                   dds_eps_rel=self.dds_eps_rel,
                                   dds_softmax_tau=self.dds_softmax_tau,
                                   dds_novelty_weight=self.dds_novelty_weight
                                   )

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data,self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self._population_management(population, self.pop_size)
                n_init_retry = 0
                while len(population) == 0 and n_init_retry < 5:
                    print(
                        "initial population has no feasible individuals; "
                        "regenerating ..."
                    )
                    population = interface_ec.population_generation()
                    population = self._population_management(population, self.pop_size)
                    n_init_retry += 1
                if len(population) == 0:
                    raise RuntimeError(
                        "Failed to initialize a feasible population after retries."
                    )

                # print(len(population))
                # if len(population)<self.pop_size:
                #     for op in [self.operators[0],self.operators[2]]:
                #         _,new_ind = interface_ec.get_algorithm(population, op)
                #         self.add2pop(population, new_ind)
                #         population = self.manage.population_management(population, self.pop_size)
                #         if len(population) >= self.pop_size:
                #             break
                #         print(len(population))
     
                
                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)
        for pop in range(n_start, self.n_pop):  
            #print(f" [{na + 1} / {self.pop_size}] ", end="|")         
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|") 
                op_w = self.operator_weights[i]
                parents = []
                offsprings = []
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                self._record_memory_improvements(parents, offsprings, population, pop + 1, op)
                self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                for off in offsprings:
                    print(" Cost(obj): ", off['objective'], end="|")
                    gap_val = self._numeric_gap(off)
                    if gap_val is not None:
                        print(" Gap(%): ", np.round(gap_val, 3), end="|")
                    t_val = self._numeric_time_cost(off)
                    if t_val is not None:
                        print(" Time(s): ", np.round(t_val, 3), end="|")
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                # populatin management
                size_act = min(len(population), self.pop_size)
                population = self._population_management(population, size_act)
                print()

            population = self._validation_select_for_next_train(population, pop + 1)

            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)


            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()
            pop_gaps = [self._numeric_gap(ind) for ind in population]
            pop_gaps = [g for g in pop_gaps if g is not None]
            if len(pop_gaps) > 0:
                print("Pop Gaps(%): ", end=" ")
                for g in pop_gaps:
                    print(str(np.round(g, 3)) + " ", end="")
                print()
