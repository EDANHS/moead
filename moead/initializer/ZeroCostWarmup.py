import time
import sys
import numpy as np
import multiprocessing
from moead.utils import zcp_evaluation_worker


class ZeroCostWarmup:

    def __init__(self, warmup_size: int = 1000, verbose: int = 1):
        self.warmup_size = warmup_size
        self.verbose = verbose

    def _force_print(self, message: str):
        sys.stdout.write(message + '\n')
        sys.stdout.flush()

    def _sanitize_metric(self, value) -> float:
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)

    def execute(self, problem, n_pop: int, history=None) -> list:

        self._force_print(
            f"\n[WARMUP] Iniciando prospección estructural multicriterio. "
            f"Muestreando {self.warmup_size} topologías..."
        )

        start_time = time.time()

        candidate_pool = []

        # ==========================================================
        # 1. Generación de candidatos
        # ==========================================================
        while len(candidate_pool) < self.warmup_size:

            sol = problem.create_solution()

            if (
                getattr(sol, '_invalid_genotype', False)
                or not problem._is_within_bounds(sol.variables)
            ):
                continue

            config = problem.decode_solution(sol.variables)

            candidate_pool.append({
                'solution': sol,
                'config': config,
                'zcp_metrics': None,
                'pred_dice': None
            })

        self._force_print(
            "[WARMUP] Orquestando procesos de evaluación con telemetría activada..."
        )

        ctx = multiprocessing.get_context("spawn")

        # ==========================================================
        # 2. Evaluación ZCP
        # ==========================================================
        for idx, candidate in enumerate(candidate_pool):

            if self.verbose >= 1 and idx % 50 == 0:
                self._force_print(
                    f"--> [PROSPECCIÓN] Progreso: "
                    f"{idx}/{self.warmup_size} arquitecturas procesadas."
                )

            start_eval = time.perf_counter()
            queue = ctx.Queue()

            process = ctx.Process(
                target=zcp_evaluation_worker,
                args=(
                    queue,
                    candidate['config'],
                    problem.input_shape,
                    problem.max_trainable_params,
                    problem.use_gpu,
                    "ensemble"
                )
            )

            process.start()

            try:

                result = queue.get(timeout=60)
                elapsed_time = time.perf_counter() - start_eval

                if result.get("success", False):

                    synflow = self._sanitize_metric(
                        result.get('zcp_synflow', 0.0)
                    )

                    snip = self._sanitize_metric(
                        result.get('zcp_snip', 0.0)
                    )

                    jacobian = self._sanitize_metric(
                        result.get('zcp_jacobian', 0.0)
                    )

                    # ==================================================
                    # Guardar SIEMPRE los valores reales
                    # ==================================================
                    zcp_metrics = {
                        'zcp_synflow': synflow,
                        'zcp_snip': snip,
                        'zcp_jacobian': jacobian
                    }

                    candidate['zcp_metrics'] = zcp_metrics

                    # Propagar a la solución para futuras evaluaciones
                    setattr(
                        candidate['solution'],
                        'zcp_metrics',
                        zcp_metrics
                    )

                    # ==================================================
                    # Predicción subrogada usando ZCP reales
                    # ==================================================
                    config = candidate['config'].copy()
                    config.update(zcp_metrics)

                    candidate['pred_dice'] = (
                        problem.surrogate.predict_loss(config)
                    )

                    raw_params = problem._calculate_params_analytical(config)

                    z_min = getattr(problem, 'z_min_params', 53)
                    z_max = getattr(problem, 'z_max_params', 35000000.0)

                    obj_params_norm = float(
                        np.clip(
                            (raw_params - z_min) / (z_max - z_min),
                            0.0,
                            1.0
                        )
                    )

                    if self.verbose >= 1:

                        self._force_print(
                            f"\n--> [CACHE MISS] Evaluando arquitectura: {config}"
                        )

                        self._force_print(
                            f"    Resultados -> "
                            f"Dice Loss Pred: {candidate['pred_dice']:.4f} | "
                            f"Params Norm: {obj_params_norm:.4f} | "
                            f"Latencia: {elapsed_time:.4f}s"
                        )

                        self._force_print(
                            f"    [OK] Arq {idx:04d} | "
                            f"ZCP-Synflow: {synflow:.2e} | "
                            f"ZCP-SNIP: {snip:.2e} | "
                            f"ZCP-Jacobian: {jacobian:.2e}"
                        )

                else:

                    if self.verbose >= 1:
                        self._force_print(
                            f"    [!] Arq {idx:04d} Fallida: "
                            f"{result.get('error', 'Desconocido')}"
                        )

            except Exception as e:

                if self.verbose >= 1:
                    self._force_print(
                        f"    [ERROR] Worker {idx:04d}: {e}"
                    )

            finally:

                process.join()

        # ==========================================================
        # 3. Selección Darwiniana
        # ==========================================================
        valid_pool = [
            c for c in candidate_pool
            if c['zcp_metrics'] is not None
            and c['pred_dice'] is not None
        ]

        if not valid_pool:

            self._force_print(
                "[CRÍTICO] Warmup: ninguna arquitectura válida."
            )

            elite_solutions = [
                c['solution']
                for c in candidate_pool[:n_pop]
            ]

        else:

            # Maximizar ZCP, minimizar Dice
            valid_pool.sort(
                key=lambda x: (
                    -x['zcp_metrics']['zcp_synflow'],
                    -x['zcp_metrics']['zcp_snip'],
                    -x['zcp_metrics']['zcp_jacobian'],
                    x['pred_dice']
                )
            )

            elite_solutions = [
                c['solution']
                for c in valid_pool[:n_pop]
            ]

        self._force_print(
            f"\n[WARMUP] Selección completada en "
            f"{time.time() - start_time:.2f}s. "
            f"{len(elite_solutions)} individuos seleccionados."
        )

        return elite_solutions