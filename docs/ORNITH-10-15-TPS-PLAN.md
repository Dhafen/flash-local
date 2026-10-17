# Ornith: arbetsplan mot 10–15 token/s på iPhone

Beslut 2026-10-17. Måltelefon: iPhone 17 Pro Max. All inferens och alla
tränade modellvikter stannar på telefonen. Processbudget högst 5,5 decimal GB,
eller lägre om iOS ger mindre utrymme. Befintliga GGUF/REAM-spår bevaras.

**10–15 token/s är ett utvecklingsmål vars genomförbarhet ännu behöver mätas.**
Godkänt resultat kräver användbar, uthållig hastighet och oförsämrad kvalitet
i en låst jämförelse. Kortvarig hastighet, färre experter eller en kortare
kontext får inte redovisas som att det ursprungliga målet är uppnått.

## 1. Utgångsläge och tidsbudget

Den fysiska rapporten från bygge 94 mäter fortfarande **GGUF**, inte den
separata Swiftlet/QPack-motorn. Identifierare: Ornith Q4_K_M, 21,713 GB,
4 096 kontextkapacitet och fälten `decodeProjections`/`packedMetalCalls`.
Appen öppnar fortfarande detta läge som standard; menyvalet för den nya
motorn är lätt att missa. Nästa användarflödesändring ska visa aktiv motor
och erbjuda ett tydligt motorval, med fortsatt separata modellinstallationer.

| Build 94, nio decode-steg | Uppmätt |
| --- | ---: |
| Decode forwards/s | 1,18396 |
| Tid per forward | 844,6 ms |
| Projektionernas `computeSeconds` per forward | 390,2 ms |
| Väntan på viktläsning per forward | 176,0 ms |
| Utvärderingsbatcher per forward | 395 |
| Returnerade läsbytes per forward | 290,6 MB |
| Prefill för 21 nya token | 22,02 s |
| Återanvända prompttoken | 31 |
| Observerad maximal process-footprint | 3,878 GB |

`computeSeconds` omfattar körning och synkronisering på värdsidan; det är
ingen separat GPU-tidsstämpel. Fälten är inte en fullständig, överlappsfri
tidsuppdelning. Returnerade läsbytes är inte fysisk NAND-trafik. Nio token
räcker inte för att fastställa uthållig hastighet. Build 84 har annan
promptlängd och okänd temperatur, så skillnaden mot 94 är inget isolerat A/B.

| Mål | Total tid per decode-steg | Krävd förbättring från mätningen | Returnerad lästakt om bytes/token är oförändrat |
| --- | ---: | ---: | ---: |
| 10 token/s | 100 ms | 8,45 gånger | 2,91 GB/s |
| 15 token/s | 66,7 ms | 12,67 gånger | 4,36 GB/s |

Detta är beräknade krav, inte påståenden om telefonens flashbandbredd.
Att eliminera enbart de uppmätta 176 ms läsväntan skulle, om allt annat
förblev lika, ge cirka 1,50 forwards/s. Både körning och väntan måste ned.
Prefill, laddning och generering ska redovisas separat; dessutom mäts tid
till första synliga token och verklig utmatningstakt inklusive löpande overhead.

Swiftlets egen README uppger ungefär 1 token/s för en 35B-modell på iPhone 17
och 7–11 på M5 Mac. Det är andra modeller/körningar, inte vårt tak eller
vårt resultat. Det ger heller inget belägg för att bara byta motor räcker.
[Källa: Swiftlet](https://github.com/leonickson1/Swiftlet).

## 2. Det vi redan återanvänder

Bygge 94 innehåller QPack-motorn under **ⓘ → Ornith QPack (testläge)**.
Den hämtar ett separat MLX-paket på cirka 19,51 GB och har 8 192 tokens
kontextkapacitet. Ingen fysisk QPack-rapport har kommit in ännu.

Den låsta runtime-versionen är `1388b08883d3d469addb5fdb88916ac7c5a19cd6`.
Källgranskningen visar att den redan har:

- 1 389 394 176 byte residenta dense-vikter, utan full dekvantisering.
- Cirka 2 GB initial expertcache med dynamisk minnesprövning, LFU och recency.
- Ett sammanhängande QPack-block per expert, direktläsning till GPU-delade
  buffertar och parallella läsningar av flera missar inom samma lager.
- GPU-baserad attention/DeltaNet och ingen oanvänd CPU-kopia av KV.
- Lagerbaserad prefill i upp till 32 tokens, expertunion per lager och exakt
  återanvändning av redan bearbetade promptprefix.
- GPU-tid där tillgänglig, anropsantal, läsbytes, cacheträffar, footprint och
  iOS termiska tillstånd i rapporten.

Följande återstår i den kod vi faktiskt kör:

- `encodePendingMoE` gör separata gate/up/down-anrop och SiLU-anrop för
  var och en av åtta experter. Expertkedjorna blir 32 dispatchar per lager,
  alltså 1 280 per token före shared expert, attention, router och övrigt.
- `stepOneFast` väntar på GPU-resultatet för routing i varje lager och
  väntar sedan på expertläsning. Det blir 40 lagerbuffertar och en slutbuffert.
  Routingberoendet är verkligt; GPU-top-k ensamt tar inte bort diskberoendet.
- Cachemissarna läses parallellt men anropet blockerar tills gruppen är klar.
  Det är ännu ingen pipeline som överlappar denna väntan med GPU-arbete.
- Cache kan krympas under körning; uppmätt säker tillväxt saknas.

Se [runtime-kod](https://github.com/Dhafen/flash-local-swiftlet/blob/1388b08883d3d469addb5fdb88916ac7c5a19cd6/Sources/SwiftletCore/QwenMetalModel.swift)
och [cache](https://github.com/Dhafen/flash-local-swiftlet/blob/1388b08883d3d469addb5fdb88916ac7c5a19cd6/Sources/SwiftletCore/ExpertCache.swift).

## 3. Arbetsordning och beslutspunkter

Varje ändring får en separat flagga/checkpoint och jämförs A/B/A på samma
telefon. Behåll bara en prestandaändring när nyttan återkommer i minst tre
parade försök, överstiger mätvariationen och numerik/minne passerar. För
små ändringar används 10 procent som praktisk granskningsgräns; en viktig
korrekthetsfix behöver naturligtvis inte ge en hastighetsvinst.

| Ordning | Konkret arbete | Vad avgör nästa steg? |
| --- | --- | --- |
| P0 | Mät befintlig QPack på telefonen, lås modell/prompt/policy, jämför numerik. | Faktisk tid i GPU, CPU/dispatch och expertfetch. Ingen hastighetsbudget fylls med antaganden. |
| P1 | Samla gate/up för alla åtta valda experter i en grid; batcha SiLU och down; behåll ordnad viktad ackumulering. | Färre dispatchar och lägre forwardtid med samma expert-ID:n och verifierade logits. |
| P2 | Överlappa exakt nödvändiga expertläsningar med shared-expert-arbete och redan cachade experter. | Minskad kritisk väntan, kontrollerade livstider och ingen extra läs-/minnesexplosion. |
| P3 | A/B-test av cachebudget, åldrande LFU/recency och eventuella lagerkvoter. | Lägre missbytes/token och stabilare tid i kod-/agentprompter, inom 5,5 GB. |
| P4 | Optimera befintlig lagerprefill med riktiga batched matmuls och chunkstorlekar. | Lägre prompttid och identisk sekvenslogik; ingen decode-vinst tillräknas prefill. |
| P5 | Pröva verifierad spekulativ avkodning om den vanliga vägen fortfarande ligger under målet. | Accepterade målmodelltoken per total sekunda, inklusive draft, verifiering och återställning. |
| P6 | Uthållighet, verklig 8K-kontext, därefter 32K och 65K; sedan LAN-API. | Separata mätresultat per kontext, säker minnesmarginal och ingen kvalitetsregression. |

### P0: mät en rättvis baslinje och stäng numerikgapet

Första telefonprovet i QPack är bara ett startprov. Nästa benchmarkläge ska
köra låsta, offentliga svenska/engelska kod- och instruktionsprompter
automatiskt och exportera modellhash, motor, policy och mätningar. Det ska
inte kräva att användaren manuellt jämför många korta chattar.

Protokoll: 256 prompttoken och minst 128 faktiska decode-steg; komplettera
med längre kodsvar, uppföljning med prefixreuse och cirka 7 900 prompttoken
plus svar inom 8 192 total kapacitet. Stopptoken respekteras. Ett tidigt EOS
redovisas som kort prov, inte som fullständigt benchmark. För varje fall:
tre kalla applikationscacher och tre varma, samma input/policy; redovisa
att iOS filcache inte går att garantera kall genom att bara starta om appen.

Mät median/p95 tokenlatens, sammanlagd decode-takt, tid till första token,
prefill-token/s, GPU-tid, encode/commit/väntetid, cacheträffar per generation,
returnerade läsbytes och läsanrop, RAM-topp, termiskt tillstånd och lågströmsläge.
Utöka rapporten med runtime-kodens befintliga `CPUGapBreakdown` samt
sammanfattad bufferttidslinje där nuvarande export saknar dessa detaljer.
GPU-tid och CPU-väntetid får inte summeras som oberoende kostnader.

Kör ett separat läsmikrobenchmark på befintliga lokala expertblock och ett
resident mikrobenchmark på en begränsad uppsättning lager. Detta avgränsar
I/O, dispatch och beräkning inom samma RAM-budget. Ett resident lagerprov
är inte bevis för att hela modellen kan köras resident eller lika snabbt.

### P1: färre GPU-anrop, samma modellberäkning

Första prototypen ska gruppera oberoende expertprojektioner utan att ändra
de enskilda dotprodukternas reduktionsordning. Nästa kandidat kombinerar
gate/up och SiLU där det går att behålla referensens avrundning. Designmål:
gå från 32 till ungefär 3–4 anrop för de åtta expertkedjorna per lager.
Detta är ett dispatchmål, inte ett löfte om motsvarande tok/s-faktor.

Använd buffer-offsettabeller/argumentbuffertar så att cacheträffar inte
kopieras till ett nytt expertpaket. Behåll alla åtta routingval och vikter.
Behåll deterministisk ackumulering efter down-projektionerna. Därefter
profilstyrda kandidater: samlade Q/K/V-projektioner, norm/residual-fusion
och mindre småanropsöverhead i DeltaNet. Varje kandidat jämförs med den
oförändrade vägen både på gränsfallsfixturer och tränade telefonlogits.

### P2–P3: läs medan GPU arbetar, behåll nyttiga experter

Efter att **riktiga** routerval är kända: reservera/pinna buffertplatser,
starta begränsade async-läsningar, kör shared expert och tillgängliga expert-
projektioner, kör resten när läsningarna är klara och ackumulera i originalordning.
Pröva liten begränsad I/O-kö och dubbel buffring; ett obegränsat antal
tasks/read-anrop skapar ingen gratis bandbredd. Logga tidslinjen så att
överlappningen kan visas, inte bara antas från summan av lästider.

Nödvändiga tester: buffert får aldrig återanvändas innan sista GPU-läsningen,
kort läsning/fel/cancel, cachekrympning, bakgrundning och samtidiga cachemissar.
Framtida förladdning får bara vara en hint: ett felgissat expertval ska
ersättas med modellens faktiska val och få sin kostnad räknad. Ingen
approximerad routing eller ersättning av missande experter.

Jämför LFU med åldrande och recency samt skydd mot att en lång prefill tränger
undan decode-arbetsmängden. Fördela cache efter verkliga lagerträffar om det
slår nuvarande globala policy. Prova större cache endast när mätt footprint,
framtida KV/SSM, Metal-allokeringar och reserv tillåter det; telefonens
fysiska 12 GB är inte vår tillåtna processbudget.

QPack har 320 valda expertblock per decode och 1 769 472 byte per block:
566,2 MB vid noll träffar, exklusive dense-vikternas GPU-trafik. Vid exempelvis
80 procent cacheträff blir expertläsningen cirka 113,2 MB/token och kräver
cirka 1,13 respektive 1,70 GB/s returnerade läsbytes vid 10 respektive 15 token/s.
**80 procent är ett räkneexempel, inte uppmätt eller garanterad träffnivå.**

### P4: promptbearbetning och lång kontext

Layer-major/chunk-32 finns redan. Mät 8/16/32 tokens med faktisk expertunion
och scratch-allokering; större batch är inte alltid snabbare eller möjlig.
Samlade matmuls ska återanvända viktläsningarna över token. DeltaNet- och
convolutionstillstånd måste fortskrida i rätt tidsordning. Jämför logits
och slutligt KV/SSM/conv-tillstånd mot sekventiell bearbetning och prefixreuse.

Nuvarande F32 KV kostar cirka 0,336 GB vid 8K, 1,342 GB vid 32K och
2,684 GB vid 65K kapacitet. Den statiska budgetberäkningen, med 100 MB
baslinje-footprint och planens reserver, lämnar vid 65K bara cirka 0,287 GB
expertcache. Det är ingen fysisk
telefonverifiering. En 32-token expertunion kan dessutom kräva upp till
256 expertplatser, cirka 453 MB. Långkontextstödet måste därför ha adaptiv
chunk/unionhantering innan det aktiveras. KV-precision får inte sänkas som
en omärkt genväg. 10–15 token/s vid kort kontext bevisar inget om 65K.

### P5: spekulativ avkodning med full verifiering

Börja om möjligt med billiga förslag från prompt/nyligen genererad kod;
senare en kompatibel draft-modell eller MTP om vikter faktiskt finns.
Alla accepterade token måste godkännas av Ornith. Greedy-fallet ska ge exakt
samma tokensekvens som den låsta målmotorn; sampling kräver korrekt
accept/reject-algoritm, inte bara jämförelse med högsta sannolikheten.
[Metodens primärkälla](https://arxiv.org/abs/2211.17192).

Den publicerade MLX-indexfilen innehåller inga MTP-vikter trots configfältet.
Detta är inte en inställning som bara kan slås på. QPack-runnern projicerar
dessutom normalt logits bara för sista prompttoken: verifieringsbatcher
behöver alla positioners logits, samt korrekt återställning av **KV, SSM,
conv och tokenposition** vid första avslag. Prissätt dessa checkpoints,
draft-vikter och arbetsbuffertar innan de tillåts i 5,5 GB.

Mät drafttid, verifieringstid, accepterad prefixlängd, rollback och extra
läsning. Nya token måste verifieras även när förslagen ser rimliga ut.
Behåll vägen bara när accepterade token per **total** tid förbättras i
representativa kod-/textuppgifter. Räkna aldrig föreslagna token som output.

## 4. Kvalitetsgrindar

1. Lås checkpoint, tokenizer, chat template, thinkingläge, sampling och
   stopplogik. Alla 40 lager och top-8 med oförändrade routervikter används.
   Ingen ny lägre kvantisering, expertpruning, tyst trunkering eller sänkt
   precision införs under beteckningen prestandaoptimering.
2. MLX-4bit är en **annan kvantisering** än nuvarande GGUF Q4_K_M.
   Installerns bytebevarande MLX→QPack bevisar inte att bytet från GGUF
   bevarar modellkvaliteten. Separat uppgiftsjämförelse krävs före standardbyte.
3. För layout/cache/scheduling krävs samma valda experter, routervikter,
   tillstånd och deterministiska resultat. För kerneländringar registreras
   absolut/relativ logitavvikelse, top-1/greedy-sekvens och lageravvikelser;
   toleranser bestäms före jämförelsen. En liten logitavvikelse räcker inte
   som allmänt kvalitetsbevis, särskilt nära routing- och argmaxgränser.
4. Integrera den separata bounded CPU-referensen på telefonen för korta
   tränade kontrollfall. Kontrollera också publisherns BF16-semantik;
   F32 CPU mot F32 GPU ensamt verifierar inte den skillnaden. Ingen full
   tränad modell laddas på utvecklingsdator eller CI.
5. Kör låsta koduppgifter med exekverbara tester, strukturerad JSON/tool-
   formatkontroll, instruktioner och resonemang i samma konfiguration före/
   efter. En verifierad regression blockerar införandet. Ett litet testset
   ger evidens för fallen, inte ett bevis för oförändrad kvalitet överallt.

## 5. När målet får markeras uppnått

- Fysisk iPhone, samma modell/policy, flygplansprov och ingen inferensnättrafik.
- Minst tre upprepade benchmarkomgångar samt minst tio minuters sammanlagd
  aktiv generering. Redovisa hela förloppet och sista fem minuter separat,
  inklusive termiskt tillstånd; behåll avbrutna/felade försök i redovisningen.
- Minst 10 verkliga utmatade token/s i det definierade decode-provet efter
  uppvärmning, inklusive löpande sampler/UI-overhead; 15 är nästa mål.
  Redovisa dessutom full begärandetid inklusive prefill. Utan tidsstämplar
  för synliga outputtoken görs endast en forward-taktclaim.
- Processen håller den dynamiska gränsen högst 5,5 GB, med marginal för
  tillfälliga buffertar och utan jetsam. Numerik- och uppgiftsgrindarna passerar.
- 8K, 32K och 65K har separata resultat. Konfigurerad kapacitet märks inte
  som utvärderad kontext, och korta/cachade upprepningar ersätter inte långa prov.

Efter P1–P3 beräknas en ny genomförbarhetsgräns från faktisk GPU-tid,
kritisk I/O-väntan och uppmätt dataöverföring. Om dessa fortfarande överskrider
100 ms/token prövas P5 och dess verkliga acceptansgrad. Om inget passerar
kraven rapporteras högsta hållbara hastighet och exakt kvarvarande kostnad.
En mindre aktiv modell eller annan telefonbudget presenteras då som ett
separat alternativ med egen kvalitetsjämförelse; det ursprungliga målet
markeras inte som uppnått genom ett dolt modellbyte.

## 6. Leverans i detta steg

Planen ovan är framtida arbete utom det uttryckligen redan implementerade.
TestFlight-appen är fortsatt bygge 94; inga nya hastighetsvinster har mätts
eller distribuerats som del av denna plan.

`scripts/analyze_ornith_performance.py` analyserar båda rapportformaten
lokalt och räknar fram målets tid/läsbudget. Utdata innehåller bara
tillåtna mätfält; chatttext och tokenlistor kopieras inte. Skriptet känner
igen motorn från rapporten, inte från byggnumret. Det hävdar aldrig
uthållig måluppfyllelse eller kvalitetsekvivalens från en enda rapport.

```sh
python scripts/analyze_ornith_performance.py /path/to/local-report.json
python -m unittest discover -s scripts/tests -p 'test_analyze_ornith_performance.py' -v
```

Nästa beslut som behöver fysisk mätning är P0: QPack i den redan levererade
appen. [P1-prototypen finns nu separat](https://github.com/Dhafen/flash-local-swiftlet/pull/2)
med flaggan avstängd som standard. Kod/testrevision
`1e348e7b61289d70ba5bde2c212bcfa7867f4323` passerade 186 Core/livscykelkontroller
och iOS-kompilering i [körning 39027165387](https://github.com/Dhafen/flash-local-swiftlet/actions/runs/39027165387).
De åtta expertkedjorna samlas till tre GPU-dispatchar; ett litet åttalagersprov
med top-8 sparade 232 dispatchar per decode med bitvis identiska logits och
samma routing. BF16/g64 provas även med Orniths faktiska projektionsstorlekar.
Detta är mekanisk/syntetisk verifiering, inte tränad modellparitet eller
telefonhastighet. Prototypen är ännu inte integrerad i appen. Ingen ny
modellhämtning behövs för varje efterföljande kernel- eller cacheändring.
