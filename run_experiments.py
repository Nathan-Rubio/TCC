"""Roda todos os experimentos da monografia e salva um JSON por configuração.

Uso:
    python run_experiments.py                # 2 segmentações x top_k 3,5,7 + baseline sem RAG
    python run_experiments.py --repeats 3    # repete cada configuração 3 vezes (média e desvio)

Saída: pasta results/ (um JSON por execução) e results/resumo.csv.
O Ollama precisa estar rodando com os modelos mxbai-embed-large e llama3.
"""
import argparse
import csv
import os
from types import SimpleNamespace

import torch

import evaluate_rag as ev
from config import CHAT_MODEL

SEGMENTACOES = {
    "automatica": "chunks.json",
    "corrigida": "final/chunks.json",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="evaluation_dataset.json")
    p.add_argument("--model", default=CHAT_MODEL)
    p.add_argument("--top_ks", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--outdir", default="results")
    p.add_argument("--sem_baseline", action="store_true", help="Não rodar o baseline sem RAG")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dataset = ev.carregar_json(args.dataset)
    linhas = []

    def registrar(nome, k, rep, metricas):
        linhas.append({"config": nome, "top_k": k, "rep": rep, **{
            m: metricas[m] for m in (
                "total_perguntas", "recall_at_k_medio", "mrr_medio",
                "keyword_recall_medio", "latencia_media_s")}})
        with open(os.path.join(args.outdir, "resumo.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)

    for nome, caminho in SEGMENTACOES.items():
        chunks = ev.carregar_json(caminho)
        emb = ev.gerar_embeddings_chunks(chunks)  # uma vez por segmentação
        for k in args.top_ks:
            for rep in range(1, args.repeats + 1):
                print(f"\n=== {nome} | top_k={k} | repetição {rep} ===")
                res, met = ev.avaliar_rag(chunks, dataset, emb, args.model, top_k=k)
                cfg = ev.montar_configuracao(SimpleNamespace(
                    model=args.model, chunks=caminho, dataset=args.dataset,
                    top_k=k, sem_rag=False), len(chunks))
                saida = os.path.join(args.outdir, f"{nome}_k{k}_r{rep}.json")
                ev.salvar_resultados(res, met, saida, cfg)
                registrar(nome, k, rep, met)

    if not args.sem_baseline:
        for rep in range(1, args.repeats + 1):
            print(f"\n=== baseline sem RAG | repetição {rep} ===")
            res, met = ev.avaliar_rag([], dataset, torch.empty(0), args.model, sem_rag=True)
            cfg = ev.montar_configuracao(SimpleNamespace(
                model=args.model, chunks=None, dataset=args.dataset,
                top_k=None, sem_rag=True), 0)
            ev.salvar_resultados(res, met, os.path.join(args.outdir, f"sem_rag_r{rep}.json"), cfg)
            registrar("sem_rag", "-", rep, met)

    print(f"\nResumo salvo em {os.path.join(args.outdir, 'resumo.csv')}")


if __name__ == "__main__":
    main()
