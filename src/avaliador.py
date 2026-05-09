from gerar_instancia import gerar_instancia


def criar_plano_exemplo_manual(instancia):
    """
    Cria um plano de exemplo MANUALMENTE, só para testar o avaliador.
    Não é otimizado — é uma forma rápida de ter um plano concreto.
    
    Estratégia simples:
    - Cada lote vai para a primeira linha que pode (L1 se possível, senão L2)
    - L0 é feito 1 ou 2 dias antes (respeitando lead time)
    - Distribui ao longo dos dias
    - Atribui operadores em ordem
    """
    
    plano = []
    operadores_disponiveis = [op["id"] for op in instancia["operadores"]]
    contador_alocacoes = 0
    
    for ref in instancia["refs"]:
        # Para cada lote desta referência
        for lote_n in range(1, ref["lotes_a_produzir"] + 1):
            
            # Decidir linha (L1 se pode, senão L2)
            if ref["pode_L1"]:
                linha = "L1"
            else:
                linha = "L2"
            
            # Decidir dia L1/L2 (alternando para distribuir carga)
            # Começamos no dia 2 para ter margem do lead time
            dia_L1L2 = ((contador_alocacoes) % (instancia["n_dias"] - 1)) + 2
            
            # Calcular dia L0 (respeitando lead time da ref)
            dia_L0 = dia_L1L2 - ref["lead_time_L0_dias"]
            
            # Atribuir operadores em ordem (rotativo pelo pool)
            ops_necessarios = ref["operadores_L1"] if linha == "L1" else ref["operadores_L2"]
            ops_atribuidos = []
            for i in range(ops_necessarios):
                idx = (contador_alocacoes + i) % len(operadores_disponiveis)
                ops_atribuidos.append(operadores_disponiveis[idx])
            
            alocacao = {
                "ref_id": ref["id"],
                "lote_num": lote_n,
                "linha": linha,
                "dia_L1L2": dia_L1L2,
                "dia_L0": dia_L0,
                "operadores": ops_atribuidos,
            }
            plano.append(alocacao)
            contador_alocacoes += 1
    
    return plano


def imprimir_plano(plano):
    """Imprime o plano de forma legível."""
    print("=" * 70)
    print("PLANO DE PRODUÇÃO")
    print("=" * 70)
    print(f"Total de alocações: {len(plano)}")
    print()
    
    for i, aloc in enumerate(plano, start=1):
        print(f"#{i:2d} {aloc['ref_id']} (lote {aloc['lote_num']}) | "
              f"L0 dia {aloc['dia_L0']} → {aloc['linha']} dia {aloc['dia_L1L2']} | "
              f"ops: {aloc['operadores']}")


# Bloco de teste
if __name__ == "__main__":
    # Gerar uma instância pequena
    instancia = gerar_instancia(n_refs=5, n_familias=3, n_dias=5, seed=42)
    
    # Criar plano manual a partir dessa instância
    plano = criar_plano_exemplo_manual(instancia)
    
    # Imprimir
    imprimir_plano(plano)