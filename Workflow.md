Qui viene illsutrato il procedimento che abbiamo attraversato per ottenere il risultato finale.
Abbiamo cominciato con la lettura approfondita di una lista di paper tra cui quello fondamentale per il progetto la lista è la seguente:
Leggere paper 
Paper base: https://stem.elearning.unipd.it/pluginfile.php/1367026/mod_folder/content/0/wang2018deep.pdf?forcedownload=1
immaginare come saranno encoder e decoder
Paper aggiuntivi (sebastiano):
https://www.mdpi.com/2078-2489/10/8/257 (5 conv e 5 deconv, based on VGG-16)
https://www.mdpi.com/2076-3417/12/1/309 (per report e evaluation measures)
https://www.sciencedirect.com/science/article/pii/S0925231222004714 (se vogliamo aggiungere transformer?)
Paper aggiuntivi (simone):
https://arxiv.org/abs/1610.01563 (Obj recog e visual saliency sono estremamente legate, quindi una rete ottima assomiglia molto ad una rete che identifica oggetti)
https://ieeexplore.ieee.org/abstract/document/10331919 (come ottimizzare le CNN per ottenere ottimi risultati a basso costo, e ottimizzare il training nel nostro caso. Possiamo contemplare un processing lineare pre inserimento nella NN per facilitare l’addestramento e magari ridurne la complessità)
https://ieeexplore.ieee.org/abstract/document/11003418?casa_token=AM5UYz_fJmIAAAAA:GkwzuDaLpXjXLUnOPk42oE-uEp2YvYBp_PwlwrrvHSrzfdO4KKgHwK795T3JrxPsNuWXj756 (come il suono influenza la visual saliency) (interessante più che utilie)

tra parentesi è possibile leggere le nostre deduzioni tra le più importanti:
- le reti per fare visual saliency sono molto simili rispetto alle reti per la obj recognition
- i transformer possono essere un ottimo passo avanti per la rete (da implementare successivamente)

Una volta finito il processo di accumulo delle informazioni siamo passati alla progettazione delle reti.

Siamo partiti da una versione Beta che copia per filo e per segno l'architettura descritta nel paper di partenza e utilizzato la MSE come loss come consigliato nella presentazione.

I risultati (soprattuto frutto del fatto che partivamo da una rete vergine a differenza del paper dove era pre trainata) sono stati abbastanza scadenti con la produzione di artefatti geometrici e smoothing tipico dell'allenamento di MSE.

Siamo quindi passati alla versione 1.0 dove abbiamo aggiornato tutte le tecniche datate come il momento di nesterov per tecniche più moderne e all'avanguardia viste durante il corso come adam e aggiunto batchNorm che nel modello originale era assente.
La seconda grande cosa che abbiamo implementato è stata una funzione di loss molto più solida tramite diverse reference che dobbiamo aggiungere (guarda le chat precedenti) dove abbiamo implementato una deep loss CC+KL+SIM

questo ha portato a degli avanzamenti molto sostanziosi che hanno portato ad un primo modello estremamente capace che produceva mappe ottime sia da un punto di vista di minimizzazione della loss che dal punto di vista soggettivo. è stato istituito anche un mini gruppo di studio con 7 persone, con il quale è stato validato in maniera abbastanza casereccia che il modello produceva saliency buone.

Arriviamo così alla 1.1 dove abbiamo scelto di non modificare l'archiettura (1.1 infatti non 2.0) dove abbiamo reso l'input rettangolare e aumentato la dimensione e implementato l'early stopping. Nonostante avessimo grosse aspettative verso questo nuovo training i risultati non hanno prodotto risultati eclatanti solo piccoli marginali miglioramenti.

Infine per produrre un progetto più avanzato e articolato siamo passati alla versione 2.0 dove abbiamo tolto il blocco encoder e lo abbiamo sostituito con un transformer pre trainato da google image net. Qui le nostre aspettative erano altissme, ma anche dopo un lunghissimo training probabilmente frutto di un basso learning rate i risultati sono bene o male gli stessi, con dei miglioramenti ancora marginali.

Per validare e comparare i modelli abbiamo messo le mani su un test set appostito MIT che abbiamo utilizzato per testare la generalizzazione della rete e paragonare v1.1 con v2.0 e i risultati sono stati abbastanza uniformi, loss medie molto simili e anche da un punto di vista visivo pressochè identiche.