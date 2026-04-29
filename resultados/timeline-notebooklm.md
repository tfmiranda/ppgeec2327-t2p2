Explique detalhadamente todas as técnicas de fine-tuning abordadas pelas fontes. Apresente-as em ordem cronológica de surgimento e evidencie para cada uma:
- As técnicas e conceitos que as inspiraram;
- Vantagens e desvantagens em relação às suas antecessoras;
- Pontos importantes para a utilização delas na prática.


A evolução das técnicas de *fine-tuning* (ajuste fino) de Large Language Models (LLMs) tem sido impulsionada por um embate contínuo entre a necessidade de adaptar os modelos a domínios específicos e os gigantescos custos computacionais, de memória e energia exigidos [1], [2]. 

Abaixo, detalho as técnicas abordadas pelas fontes, organizadas de forma cronológica com base no seu surgimento e no avanço do estado da arte:

### 1. Full Fine-Tuning (FFT) / Ajuste Fino Completo
*   **Inspiração:** Baseado no conceito tradicional de *transfer learning* (aprendizado por transferência) do Deep Learning, onde um modelo é primeiramente treinado em dados gerais e, em seguida, todos os seus parâmetros são recalibrados para uma tarefa ou conjunto de dados específico [3], [1].
*   **Vantagens/Desvantagens:**
    *   *Vantagens:* Permite mudanças comportamentais profundas e normalmente atinge o desempenho máximo e a melhor precisão na tarefa alvo [4], [5].
    *   *Desvantagens:* Inviável para LLMs com bilhões de parâmetros em hardwares comuns, pois exige altíssimo poder computacional e uso massivo de memória para armazenar os estados do otimizador [6], [1]. Além disso, é altamente suscetível ao "esquecimento catastrófico" (*catastrophic forgetting*), onde o modelo perde o conhecimento geral previamente adquirido [4].
*   **Utilização na Prática:** Geralmente reservado para modelos menores (ex: BERT, RoBERTa) ou quando a corporação dispõe de clusters computacionais de alto nível e há necessidade de adaptação total a uma linguagem muito específica [7], [4].

### 2. Adapters / Camadas Adaptadoras (Houlsby et al., 2019)
*   **Inspiração:** Inspirado na inserção de estruturas com arquitetura de gargalo (*bottleneck*). Em vez de modificar o LLM inteiro, inserem-se novos e pequenos módulos sequenciais após as camadas de atenção e as camadas feed-forward (MLP) do Transformer [1], [8], [9].
*   **Vantagens/Desvantagens em relação ao FFT:**
    *   *Vantagens:* Muito eficiente parametralmente, sintonizando uma fração mínima do modelo (geralmente entre 0,5% e 8%). Isso acelera o treinamento em até 60% e aumenta a robustez, prevenindo o esquecimento catastrófico [9], [10], [11].
    *   *Desvantagens:* Adiciona latência extra na inferência porque o fluxo de dados na rede se torna mais longo (as ativações precisam passar pelas novas camadas sequenciais) [12], [13].
*   **Utilização na Prática:** Excelente para *Multi-task Learning*. Múltiplos "adapters" treinados podem ser acoplados a um modelo simultaneamente para que ele lide com diferentes domínios, tornando-o altamente modular [14], [11].

### 3. Prefix-Tuning (Li & Liang, 2021)
*   **Inspiração:** Derivado do conceito de *prompting*, mas expandido. Introduz e otimiza "tokens contínuos" ou vetores prefixados diretamente nos estados ocultos em *todas* as camadas do Transformer [1], [15].
*   **Vantagens/Desvantagens em relação aos Adapters:**
    *   *Vantagens:* Altamente eficiente sem alterar a estrutura da rede com novas subcamadas estruturais, focado apenas nas ativações computadas [15], [16].
    *   *Desvantagens:* Diminui o comprimento de sequência que o usuário pode utilizar (pois os prefixos ocupam espaço do contexto); além disso, sua otimização e convergência costumam ser mais instáveis e difíceis em comparação ao FFT [17], [18], [19].
*   **Utilização na Prática:** Muito usado em modelos autorregressivos (como GPT) ou encoder-decoder para tarefas de geração e compreensão natural de textos, mas sensível à sua parametrização inicial [15].

### 4. Prompt Tuning (Lester et al., 2021) / Soft Prompting
*   **Inspiração:** Uma simplificação extrema do Prefix-Tuning. O Prompt Tuning não altera nem o modelo e nem as camadas intermediárias, apenas adiciona vetores aprendíveis (os *soft prompts*) à camada de *input embeddings* (entrada) [1], [20], [21].
*   **Vantagens/Desvantagens em relação ao Prefix-Tuning e Adapters:**
    *   *Vantagens:* É o método de implementação mais simples. Requer um custo quase nulo de armazenamento, sendo excelente para aplicações Zero-Shot e Few-Shot [20], [22].
    *   *Desvantagens:* Sofre com lentidão na convergência (Maldição do Treinamento) e seu desempenho é muito ruim para modelos menores. No entanto, exibe a propriedade do "Poder da Escala", alcançando os resultados do FFT somente quando aplicado a modelos massivos (acima de 10 bilhões de parâmetros) [20], [17], [23].
*   **Utilização na Prática:** Ideal para situações onde a organização não possui muitos dados ou não quer alterar *nenhum* dos pesos do modelo para preservar o comportamento original de forma estrita, dependendo de modelos muito grandes [24], [25].

### 5. BitFit (Ben Zaken et al., 2021)
*   **Inspiração:** Abordagem de especificação heurística. A hipótese de que o ajuste em redes grandes tem uma dimensionalidade intrínseca tão baixa que apenas sintonizar uma parte isolada específica do modelo seria suficiente [26], [27].
*   **Vantagens/Desvantagens em relação aos modelos acima:**
    *   *Vantagens:* Simplicidade brutal. Ele atua congelando o modelo inteiro e treinando *apenas* os termos de viés (*bias terms*). Envolve menos de 0,1% dos parâmetros do modelo [26], [28].
    *   *Desvantagens:* Apesar de performar incrivelmente bem em modelos pequenos no benchmark GLUE, não há garantias de manter sua competitividade com abordagens em modelos com bilhões de parâmetros ou tarefas complexas [26].
*   **Utilização na Prática:** Ótima estratégia inicial e de baixo custo, especialmente quando dados de treinamento estão escassos, porém, tem utilidade reduzida à medida que as necessidades do domínio se distanciam dos dados originais [26].

### 6. LoRA - Low-Rank Adaptation (Hu et al., 2021)
*   **Inspiração:** Baseia-se na teoria da "dimensão intrínseca". O LoRA pressupõe que as matrizes de peso das redes neurais mudam em direções que requerem baixo "posto" (Low-Rank) matemático durante o fine-tuning. Assim, o modelo original é congelado e a aprendizagem ocorre em duas matrizes de decomposição de baixo posto ($A$ e $B$) injetadas nos módulos do Transformer [1], [27], [29], [30].
*   **Vantagens/Desvantagens em relação aos Adapters e Prompt Tuning:**
    *   *Vantagens:* **Não possui a latência de inferência** característica dos Adapters (pois as matrizes $A$ e $B$ são unidas/fundidas aos pesos originais antes de ser utilizado). Seu desempenho frequentemente se iguala ao do Full Fine-Tuning, economizando substancial memória [12], [30], [31].
    *   *Desvantagens:* É bastante sensível a hiperparâmetros (especificamente os valores de rank $r$ e escalar $\alpha$), e pode falhar em adaptações cujas tarefas exijam repaginação maciça da capacidade de representação do modelo original [30], [32].
*   **Utilização na Prática:** É a principal técnica moderna (*State-of-the-Art* para PEFT). Normalmente aplicado apenas às matrizes projetoras de *Query* e *Value* nas camadas de atenção [33], [16].

### 7. QLoRA (Dettmers et al., 2023)
*   **Inspiração:** Unir o método LoRA à **quantização**. A quantização foca na diminuição da precisão numérica em bits dos parâmetros das redes neurais com o intuito de diminuir o seu espaço de memória [34], [35], [36].
*   **Vantagens/Desvantagens em relação ao LoRA base:**
    *   *Vantagens:* Utiliza o tipo de dado NormalFloat de 4-bits e quantização dupla. Essa economia extrema possibilita treinar LLMs maciços em placas de vídeo convencionais de consumidores, abrindo portas e popularizando o desenvolvimento IA [34], [37].
    *   *Desvantagens:* Custo moderado na velocidade do treinamento (visto a conversão computacional entre 4-bits e 16-bits constantemente). A inferência quantizada pode se mostrar levemente menos performática caso haja compressão muito agressiva [34], [38].
*   **Utilização na Prática:** Essencial quando se trabalha em hardwares com recursos limitados de RAM na GPU, frequentemente utilizado como padrão na comunidade *open-source* com o módulo *BitsAndBytes* [34], [37].

### 8. DoRA - Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
*   **Inspiração:** Observação detalhada das dinâmicas do peso no Full Fine-Tuning vs. LoRA. O DoRA aborda a aprendizagem decompondo os parâmetros em **direção** e **magnitude**. Ele aplica o Low-Rank (estilo LoRA) na parte de atualização direcional enquanto aprende um vetor isolado de magnitude [39], [40].
*   **Vantagens/Desvantagens em relação ao LoRA:**
    *   *Vantagens:* Imita de forma muito mais precisa as características do Full Fine-Tuning, mas mantendo a leveza do PEFT. Consistentemente, o DoRA supera a precisão do LoRA tradicional, mantendo-se perfeitamente mesclável ao modelo pré-treinado sem aumento de latência [39], [41].
    *   *Desvantagens:* Introduz complexidade matemática suplementar no cálculo das normalizações, o que requer uma pequena calibragem na implementação [40], [41].
*   **Utilização na Prática:** Aplicado em tarefas muito exigentes onde o modelo subjacente via LoRA costuma parar de progredir de maneira prematura [39].

### 9. Half Fine-Tuning (HFT) (Hui et al., 2024)
*   **Inspiração:** Técnica híbrida entre preservar conhecimento e reter agilidade. Foi desenhada para mitigar tanto os gargalos do FFT quanto o viés restrito do PEFT, combatendo o "esquecimento catastrófico" no processo contínuo de aprendizagem [42].
*   **Vantagens/Desvantagens:**
    *   *Vantagens:* Funciona dividindo os parâmetros do modelo e congelando metade das informações por rodada e atualizando a outra metade. Ele recupera parte dos pesos pré-treinados, sendo muito simples de aplicar, dispensando inserção de matrizes novas como o LoRA [42], [43], [44].
    *   *Desvantagens:* Exige cuidado no gerenciamento de parâmetros (ativos x inativos) em cada bloco para não distorcer as dinâmicas de pesos em redes já sintonizadas [42].
*   **Utilização na Prática:** Excelente para ambientes que requerem sintonias constantes (*continual learning*) onde reter as habilidades nativas do LLM é crítico [42].

### 10. Otimização por Preferências e Alinhamento: PPO, DPO e ORPO (2017 a 2024)
*   **Inspiração:** Todas essas são técnicas focadas no alinhamento humano de IA baseadas em recompensas (RLHF), após o processo de Fine-Tuning Supervisionado (SFT), garantindo não apenas utilidade de conteúdo, mas adequação do comportamento conversacional [45], [46].
*   **Progressão e Comparação:**
    *   **PPO (Proximal Policy Optimization, ~2017 adaptado para LLMs):** Baseado em Aprendizado por Reforço, treina um classificador separado ("Modelo de Recompensa") com *feedbacks* humanos para maximizar ganhos mediante simulação das ações do LLM. *Vantagens/Desv:* Estável na política, mas altamente complexo de configurar, intensivo em computação e suscetível à instabilidade em treinamento [47], [46].
    *   **DPO (Direct Preference Optimization, 2024):** Elimina completamente a necessidade do oneroso "Modelo de Recompensa" de reforço. Ele cria um objetivo direto de classificação com máxima verossimilhança onde o LLM é sintonizado ajustando suas gerações baseadas em um log de "respostas preferidas x respostas rejeitadas". *Vantagem:* Muito mais prático, eficiente e menos propenso à degeneração (instabilidade) [46], [48].
    *   **ORPO (Odds-Ratio Preference Optimization, 2024):** Mais inovador ainda, retira o modelo de "referência" obrigatório do DPO e age fundindo a fase de SFT com a de alinhamento em uma única perda monolítica baseada na "Razão de Chances" (*Odds-Ratio*), penalizando resultados malquistos diretamente durante a aprendizagem [49], [50].
*   **Utilização na Prática:** Utilizados ativamente no treinamento final dos modelos que interagem na modalidade *Chat*, evitando "alucinações" e respostas tóxicas ou antiéticas [46], [49].