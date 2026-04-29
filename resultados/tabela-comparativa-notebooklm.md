Resuma os métodos de fine tuning e a relação entre comparativa entre eles. Sumarize isso em formato de tabela comparativa.

O fine-tuning (ajuste fino) é o processo de adaptar um Modelo de Linguagem Grande (LLM) pré-treinado para tarefas ou domínios específicos [1, 2]. Devido ao tamanho colossal dos modelos modernos (com bilhões de parâmetros), o ajuste fino de todos os parâmetros tornou-se computacionalmente inviável, impulsionando o desenvolvimento de técnicas de **PEFT (Parameter-Efficient Fine-Tuning)** [3-5]. 

Abaixo, resumo os principais métodos, a relação comparativa entre eles e, em seguida, apresento uma tabela comparativa.

### Resumo dos Principais Métodos de Fine-Tuning

**1. Full Fine-Tuning (Ajuste Fino Completo)**
Atualiza todos os parâmetros originais do modelo [6, 7]. Embora geralmente alcance o desempenho máximo, exige enorme capacidade de memória e poder computacional [8]. Além disso, é mais suscetível ao "esquecimento catastrófico" (perda do conhecimento previamente aprendido) e a problemas de implantação, pois exige o armazenamento de uma cópia completa do modelo para cada tarefa [5, 6, 9].

**2. Métodos Baseados em Adição (Addition-based)**
Introduzem novos parâmetros ou módulos treináveis, mantendo os pesos originais congelados [10, 11].
*   **Adapters:** Inserem pequenas camadas de rede neural (adaptadores) dentro da arquitetura do Transformer (geralmente após a atenção e feed-forward) [11, 12]. Eles reduzem os parâmetros treináveis (usando cerca de 0,5% a 8% do modelo) [13], mas **adicionam latência na inferência**, pois os dados precisam passar sequencialmente por essas novas camadas [14].
*   **Prompt Tuning e Prefix-Tuning:** Adicionam tokens contínuos (prompts ou prefixos) treináveis à entrada ou aos estados ocultos do modelo, sem alterar a arquitetura interna [15, 16]. São extremamente eficientes em parâmetros, mas o *Prompt Tuning* costuma ter uma convergência mais lenta, sendo mais difícil de otimizar em modelos menores [17].

**3. Métodos Baseados em Reparametrização (Reparameterization-based)**
Transformam os parâmetros a serem ajustados em formas de menor dimensão, partindo da hipótese de que a adaptação a novas tarefas ocorre em um "subespaço de baixa dimensão" intrínseco [18, 19].
*   **LoRA (Low-Rank Adaptation):** Congela os pesos originais e introduz duas matrizes de posto baixo (A e B) para aprender as atualizações dos pesos (geralmente nas matrizes de atenção) [20-22]. A grande vantagem do LoRA é que, durante a inferência, essas matrizes podem ser mescladas com os pesos originais, resultando em **zero latência adicional** [23].
*   **QLoRA (Quantized LoRA):** Estende o LoRA usando quantização (ex: 4-bits) nos pesos congelados do modelo original [24, 25]. Isso reduz drasticamente o uso de memória (permitindo fine-tuning em GPUs únicas) com uma perda mínima de informação [24, 25].
*   **DoRA (Weight-Decomposed Low-Rank Adaptation):** Decompõe a atualização dos pesos em dois componentes: magnitude e direção [8]. Ele usa a eficiência do LoRA para otimizar apenas o componente direcional, imitando os padrões de aprendizado do *Full Fine-Tuning*. Como resultado, o DoRA frequentemente supera o LoRA em desempenho [26, 27].

**4. Métodos Baseados em Especificação / Seletivos (Specification-based)**
Ajustam apenas um subconjunto restrito de parâmetros já existentes no modelo [10].
*   **BitFit:** Treina exclusivamente os termos de viés (bias) do modelo, congelando o resto. É muito leve e atinge desempenho surpreendente em várias tarefas [7, 28].
*   **Half Fine-Tuning (HFT):** Congela metade dos parâmetros do modelo a cada ciclo de treino, atualizando a outra metade [29]. Isso não altera a arquitetura e ajuda na retenção de conhecimento fundacional (reduzindo o esquecimento catastrófico) com melhor eficiência do que o Full FT [30, 31].

---

### Tabela Comparativa de Métodos de Fine-Tuning

| Método | Custos Computacionais / Memória | Latência na Inferência / Mudança de Arquitetura | Parâmetros Treináveis (%) | Características e Relação Comparativa |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning (FT)** | **Muito Altos** [32, 33] | Nenhuma / Nenhuma | 100% [34] | **O padrão-ouro em eficácia**, mas com alto risco de *overfitting* (esquecimento catastrófico) e custo impraticável para modelos na escala de bilhões de parâmetros [4, 8]. |
| **Adapters** | **Baixos** [11, 12] | **Aumentada** (devido ao processamento sequencial) / Adição de módulos [12, 14] | ~0,5% a 8% [13] | Ótimo para aprendizagem multi-tarefa, mas, em comparação ao LoRA, perde vantagem na velocidade de inferência por aumentar o fluxo de dados (caminho mais longo) [14, 35]. |
| **Prompt Tuning** | **Muito Baixos** [36] | Nenhuma / Modifica a entrada (anexa tokens) [16, 37] | Altíssima eficiência (<0.1%) [38] | Em comparação a outros PEFTs, é o **mais fácil de implementar**, porém **converge lentamente** e tem pior desempenho em modelos menores em relação ao LoRA ou Adapters [17]. |
| **LoRA** | **Baixos** [22, 39] | **Nenhuma** (os pesos são mesclados) / Nenhuma mudança fixa [23, 27] | ~0,1% a 2% [38, 40] | Muito superior ao Prompt Tuning em estabilidade. Oferece desempenho **comparável ao Full FT**, mas evita latência na inferência e permite trocar adaptadores para tarefas diferentes facilmente [22, 41]. |
| **QLoRA** | **Mínimos** [24, 25] | Nenhuma / Usa pesos quantizados em 4-bit [24, 25] | Similar ao LoRA | **Evolução do LoRA**, focada na democratização: permite treinar LLMs gigantes em hardware modesto (ex: GPUs comerciais), ao custo de leve perda de informação [24, 25]. |
| **DoRA** | **Baixos** [8, 27] | Nenhuma / Separa a magnitude e direção dos pesos [8] | Similar ao LoRA | Supera o LoRA e **aproxima-se ainda mais do Full FT**. Entende melhor o padrão de otimização de pesos originais através de sua decomposição estrutural [26, 27]. |
| **HFT (Half Fine-Tuning)** | **Médios** [29] | Nenhuma / Nenhuma [29, 31] | ~50% [29] | Oferece um meio termo: treina muito mais parâmetros que o LoRA, porém preserva o conhecimento da fase de pré-treino muito melhor que o Full FT [9, 30]. |
| **BitFit** | **Muito Baixos** [28] | Nenhuma / Nenhuma [28] | Muito baixa | Abordagem heurística que **altera apenas o "bias"**. Pode restaurar até 95% do desempenho em tarefas simples, porém pode faltar expressividade em tarefas muito complexas ou de raciocínio lógico [28, 42]. |