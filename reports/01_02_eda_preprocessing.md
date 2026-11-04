# Exploração e pré-processamento dos dados

**Notebooks:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) e [`notebooks/02_preprocessing.ipynb`](../notebooks/02_preprocessing.ipynb)
**Dados:** CirCor DigiScope v1.0.3, parte pública: 942 pacientes, 3163 gravações

---

## Parte 1 — Exploração dos dados

Antes de desenhar os modelos, precisamos de perceber que dados temos, como estão organizados e o que é que eles nos dizem sobre o problema.
A exploração segue sempre o mesmo padrão: olhar para os dados, medir alguma coisa e tirar daí uma decisão para o pré-processamento.

### Que dados existem

O dataset tem três tipos de informação:

- **Áudio:** um ficheiro `.wav` por gravação, a 4 kHz, mono, com 5 a 65 segundos. Cada paciente tem em geral quatro gravações, uma por foco de auscultação (aórtico, pulmonar, tricúspide e mitral).
- **Anotações:** um `.tsv` por gravação, com a segmentação manual do ciclo cardíaco (S1, sístole, S2, diástole), e um `.hea` com o cabeçalho técnico do áudio.
- **Metadados:** o `training_data.csv`, com uma linha por paciente: o rótulo `Murmur`, dados demográficos (idade, sexo, altura, peso), a descrição clínica do sopro e o diagnóstico final (`Outcome`).

Verificámos que os ficheiros são consistentes entre si: cada gravação tem os seus três ficheiros, cada paciente do CSV tem gravações e os checksums oficiais batem certo nos 10 434 ficheiros.

Um ponto crítico: **só o fonocardiograma pode entrar nos modelos**. O enunciado pede classificadores construídos a partir do som, e os dados escondidos do bónus só trazem gravações.

- A descrição do sopro (fase do ciclo, forma, grau) só existe quando há sopro. Usá-la seria dar a resposta ao modelo.
- O `Outcome` é outro diagnóstico, muito associado ao `Murmur`: 150 dos 179 pacientes com sopro são *Abnormal*.
- As segmentações e os dados demográficos servem para perceber o problema, mas não entram nos modelos.

### Os rótulos e a métrica

O rótulo é do **paciente** e tem três classes: 74 % *Absent*, 19 % *Present* e 7 % *Unknown*.
A métrica é a *weighted accuracy*, em que cada *Present* conta 5 vezes, cada *Unknown* 3 vezes e cada *Absent* 1 vez.
Isto inverte a lógica habitual:

- Prever sempre *Absent* dá uma accuracy normal de 74 %, mas uma weighted accuracy de apenas 0.39.
- Prever sempre *Present* dá 0.50. **Este é o valor mínimo que um modelo tem de bater.**

Falhar um sopro custa muito mais do que um falso alarme. Por isso vamos precisar de pesos de classe no treino e de limiares de decisão afinados.

### Como é um sopro

O CSV descreve cada sopro. Quase todos são **sistólicos** (178 de 179), ocupam toda a sístole e são **muito fracos** (a maioria de grau I/VI).
Medimos isto também diretamente no sinal, com as segmentações: nas gravações com sopro, a energia acima de 150 Hz é cerca de **5 dB maior na sístole do que na diástole**; sem sopro, as duas são iguais.

A consequência é que o modelo tem de ver **ciclos cardíacos completos**, com resolução temporal suficiente (cerca de 10 ms) para separar S1, sístole e S2.

### O sopro não se ouve em todos os focos

Num paciente com sopro, o sopro pode ouvir-se só em alguns focos. **19 % das gravações de pacientes *Present* não têm sopro audível naquele foco.**
Se cada gravação herdasse o rótulo do paciente, a rede seria treinada a dizer "sopro" em gravações onde não há sopro nenhum.
Usamos o campo `Murmur locations` para criar um **rótulo por gravação**, que só serve para o treino. A avaliação é sempre feita por paciente: basta um foco com sopro para o paciente ser *Present*.

### Duração, frequência e qualidade

- **Duração:** as gravações vão de 5 a 65 s (mediana 21 s). Cortar tudo para um tamanho único deitaria fora a maior parte do sinal.
- **Batimento cardíaco:** mediana de 103 bpm e mínimo de 46 bpm. Uma janela de 4 s contém cerca de 7 ciclos em média e pelo menos 3 no caso mais lento.
- **Frequências:** S1 e S2 estão abaixo de ~150 Hz e os sopros chegam a algumas centenas de Hz. Acima de 1 kHz há sobretudo ruído (fricção, fala, choro).
- **A classe *Unknown*:** são as gravações menos periódicas (o ciclo cardíaco quase não se distingue) e com o espectro mais deslocado para frequências altas, ou seja, com mais ruído. São também muito mais comuns em bebés. *Unknown* é sobretudo uma classe de "má gravação", e vai ser a mais difícil.

### A divisão dos dados

Há 140 pacientes que são, na verdade, 70 crianças vistas nas duas campanhas de recolha. As duas visitas ficam sempre no mesmo fold.
Se uma visita estivesse no treino e a outra na validação, o modelo podia reconhecer a criança em vez de detetar o sopro, e os resultados ficariam artificialmente altos.
Os dados dividem-se por paciente: 20 % para teste, usado só no fim, e 5 folds estratificados nos restantes 80 % para escolher modelos, *early stopping* e limiares.

---

## Parte 2 — Pré-processamento

Antes de treinar os modelos, precisamos de transformar os ficheiros de áudio em dados que as redes neuronais consigam processar.
O pré-processamento divide-se em três partes principais: padronizar o áudio, extrair as entradas de cada rede e preparar os rótulos e a normalização.

### Padronização do áudio

Primeiro, padronizamos todos os áudios:

- **Mantemos os 4 kHz originais**, porque todas as gravações já estão a essa frequência e a informação útil está bem abaixo dos 2 kHz que ela permite representar.
- **Aplicamos um filtro passa-banda de 25 a 1000 Hz**. Abaixo de 25 Hz há deriva e movimento do corpo, e acima de 1000 Hz quase só há ruído.
- **Normalizamos a amplitude de cada gravação** (z-score), porque o volume varia muito entre gravações e estetoscópios.
- **Cortamos cada gravação em janelas de 4 segundos, com passo de 2 segundos**. Cada janela tem vários ciclos cardíacos completos, e todas as entradas ficam com o mesmo tamanho. O fim de cada gravação (menos de 2 s) é descartado.

Assim obtemos 31 409 janelas a partir das 3163 gravações, cerca de 10 por gravação. Cada janela herda o rótulo da sua gravação.

### Entrada da MLP

Para a MLP, cada janela é resumida num **vetor de 99 features**:

- **MFCCs** (*Mel-Frequency Cepstral Coefficients*), uma representação compacta da forma do espectro. Calculamos 20 coeficientes ao longo da janela e, como a MLP precisa de um vetor fixo, guardamos a média, o desvio padrão e o desvio padrão das variações (delta e delta-delta) de cada um: 80 valores.
- **Descritores espectrais** (centroide, largura de banda, *roll-off*, *flatness*, *zero-crossing rate* e energia), com média e desvio padrão: 12 valores. Os sopros acrescentam energia de banda larga entre S1 e S2, e estes descritores captam isso.
- **Energia relativa em 5 bandas de frequência**, entre 25 e 1000 Hz: 5 valores.
- **Periodicidade do envelope**, que mede o quanto o sinal se repete a cada batimento: 2 valores. Ajuda a reconhecer as gravações ruidosas da classe *Unknown*.

### Entrada da CNN

Para a CNN, usamos **mel-espectrogramas em escala logarítmica (dB)**: uma representação visual do áudio, em que o eixo X é o tempo, o Y é a frequência e a intensidade representa a energia.

- Cada janela dá um espectrograma de **40 bandas mel × 401 frames**: uma frame a cada 10 ms, com frequências entre 20 e 1000 Hz.
- A resolução de 10 ms permite ver S1, sístole e S2 separados, que é onde está o sopro.
- **Não normalizamos cada janela entre 0 e 1.** O nível absoluto de energia é informativo: ajuda a distinguir as gravações ruidosas.
- No treino, cada janela pode ser ligeiramente deslocada no tempo e receber máscaras de tempo e frequência (*SpecAugment*). Isto aumenta a variedade dos dados sem gravar nada de novo.

### Rótulos e normalização

Os rótulos vêm do campo `Murmur` do CSV, corrigidos por foco com `Murmur locations`, e são convertidos em índices de classe (*Present* = 0, *Unknown* = 1, *Absent* = 2).
Como as classes estão desequilibradas, a função de perda usa **pesos de classe** inversamente proporcionais à frequência de cada uma.

Um ponto crítico: **a normalização é feita apenas dentro de cada fold, usando exclusivamente os dados de treino.**

- Para a MLP, cada feature é normalizada com a média e o desvio padrão das janelas de treino.
- Para a CNN, cada banda mel é normalizada com a média e o desvio padrão das gravações de treino.

Isto evita *data leakage*. Se normalizássemos com estatísticas globais, estaríamos a "mostrar" informação da validação e do teste ao modelo durante o treino, inflacionando artificialmente os resultados.
Por isso os ficheiros gravados não têm nenhuma normalização ajustada aos dados, e podem ser gerados uma só vez para todos os folds.

### De volta ao paciente

As redes classificam janelas, mas o enunciado pede **uma previsão por paciente**:

1. As probabilidades das janelas fazem-se a média por gravação.
2. Para cada paciente, guardamos a maior probabilidade de *Present* e de *Unknown* entre as suas gravações, porque basta um foco com sopro.
3. Uma regra de decisão com dois limiares, afinados na validação, decide entre *Present*, *Unknown* e *Absent*.

### Ficheiros gerados

No final, o pré-processamento gera 5 ficheiros em `data/processed/`, prontos para o treino dos modelos:

- `X_mlp.npy`: as 99 features de cada janela (31 409 × 99), entrada da MLP;
- `mel.npy`: os mel-espectrogramas de todas as gravações, de onde as janelas da CNN (40 × 401) são cortadas durante o treino;
- `windows.csv`: uma linha por janela, com a gravação, o paciente, o foco, os rótulos e o fold;
- `recordings.csv`: uma linha por gravação, com a sua posição em `mel.npy`;
- `config.json`: todos os parâmetros do pré-processamento e os nomes das features.

---

## Conclusões

1. **Só o fonocardiograma entra nos modelos.** As outras colunas são o rótulo, descrições do próprio sopro ou informação que não existe nos dados escondidos.
2. **Desequilíbrio forte, mas a métrica favorece encontrar sopros.** Pesos de classe e limiares afinados, com o valor mínimo a bater de 0.50.
3. **Rótulos por gravação e avaliação por paciente**, porque 19 % das gravações de pacientes com sopro não o têm.
4. **Janelas de 4 s, filtro de 25–1000 Hz e resolução de 10 ms**, para ver ciclos completos onde o sopro aparece.
5. **Normalização só com os dados de treino de cada fold**, para os resultados não ficarem inflacionados.
