'''
Dette scipt laver randomsearch på de 3 .update_recognizer hyper parametre.
Sciptet kan også bruges til at teste en en bestemt configuation af parameterene ved at kalde objective_function() manuelt.
For at scriptet kan køre skal man adgang til de til transkriptionerne hørende lydklip. disse findes i zip mappen
random_search_data.zip

'''









from danspeech import Recognizer
from danspeech.pretrained_models import TestModel
from danspeech.pretrained_models import DanSpeechPrimary
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio
import librosa
import soundfile as sf
from pydub import AudioSegment
import regex
import os
import jiwer
import numpy as np
import warnings

warnings.filterwarnings("ignore")


mareridt = ["kapitel tre Nanna vågnede sent om natten, badet i sved. Hun havde en rædsel i hele kroppen. Også selvom hun ikke kunne huske, at hun havde drømt.",
           " Der var indelukket i værelset. Hun følte, hun var ved at kvæles.Det bryllup kommer til at tage livet af mig, mumlede hun for sig selv. ",
           "Hun rejste sig, åbnede vinduerne og lod den lune natte-luft strømme ind. så skidt da med myggene. ",
           "Det er alligevel bedre end at blive kvalt, tænkte hun. Hun gik ud på badeværelset og tog et iskoldt bad. ",
           "Bagefter kunne hun ikke falde i søvn igen. Hun var både sulten og tørstig. "]
           # "Den havde også en hæslig nat-hue. Hun kunne godt se, at det var hånd-syet."]
           # "Det var af langt bedre kvalitet end det, hun havde haft sidste gang. Luna havde virkelig gjort meget ud af det. Hun havde skaffet hende det bedste værelse. Helt billigt havde det nok ikke været. Hun havde også kæmpet for, at hun fik en normal madras. Hun prøvede virkelig. Åhr, så pyt da. Når nu det betyder så meget for hende, tænkte Nanna opgivende for sig selv. Hun tog både natkjole og nathue på. Så gik hun ud af værelset og ned til køkkenet. På jagt efter mad. Alle andre sov. Hun så ingen grund til at vække nogen. Hun ledte efter mad i skabene og fandt et groft brød og en mærkelig pølse. Selvfølgelig har Suzanne kun middel-alder-mad. Det må så gå, tænkte hun ærgerligt. Hun satte sig for at spise i mørket. Månen lyste fint op, og hun havde bestemt ikke lyst til selskab. Hun sad helt fordybet i sine tanker, da døren pludselig gik op. Lyset blev tændt. hun kom med et forskrækket hyl og tabte sit brød, da en mand trådte ind. Hvad satan!? sagde han, også forskrækket, og stoppede op. De stirrede på hinanden. Han var på alder med hende. Han havde kun et par cowboy-bukser på. Han var usædvanlig flot. Høj, slank og mørkhåret. Der er da utroligt, sagde han surt. Alle vegne er der tosser. I nathuer og spøgelses-kjoler. De gemmer sig endda i mørket og spiser pølse! I det mindste har jeg tøj på! svarede hun rasende. Han vendte sig og gik ud af døren. Hun hørte ham sige: Det er under al kritik, det her. I morgen rejser jeg. Hvorfor skrider du ikke bare nu? råbte Nanna efter ham. Han smækkede døren bag sig. Nanna sad tilbage og rystede af raseri. Mage til uforskammet idiot. Hun valgte at overse, at hun sådan set var enig med ham. Først da solen brød frem, rejste hun sig og gik ovenpå. Hun var alt for vågen til at sove. Hun tog natkjolen af og overvejede et øjeblik at tage almindeligt tøj på. Men så huskede hun mandens ord og blev vred igen. Med hidsige bevægelser tog hun en middel-alder-kjole på. Så gik hun ned for at finde sin søster."

anitas_millioner = ["kapitel tretten De lyse tegl-tage fortsatte, så langt øjet rakte. Langt ude kunne man se bjergene i en dis. " ,
                   "Og tæt på stak Peterskirkens kuppel op, grå og lyseblå. Anita vendte sig væk fra ræk-værket, men et råb nede fra gaden fik hende til at standse. ",
                   "Det var spå-kvinden. Hende, der havde læst i hendes hånd engang. Hun var ved at sætte et bord op og vinkede til Anita. " ,
                   "Anita vinkede tilbage. Riturnera signora. havde hun sagt. De vil vende tilbage. Og sådan var det gået. Hun var vendt tilbage. ",
                   "Hun sukkede lykkeligt og blev et øjeblik stående og så sig om. Planterne i krukkerne så allerede ud til at trænge til vand. "]
                   # "Hun havde ellers vandet dem i morges. Varmen kom tidligt i år. De var kun i maj måned, men temperaturen var allerede oppe på 27 grader. Hvor varmt ville der ikke blive på tag-terrassen, når det blev rigtig sommer? Nå, til den tid ville vindruerne give skygge. De voksede hurtigt langs med snorene, der var spændt ud over terrassen. Hun gik hen til figen-træet, der stod i sin balje. Undersøgte  de små, hårde grønne frugter. De så lovende ud. Så gik hun hen til bordet, hvor der lå en bog: Italiensk for begyndere. Ved siden af stod hendes kaffekop. Hun satte sig og slog op på det sted, hun var kommet til i bogen. Men så fik hun øje på citronerne. Det så ud til, at de var vokset i løbet af natten. Dag for dag blev de større og mere gule. Og henne i hjørnet stod oliventræerne med frugter, der knap var til at få øje på endnu. Hun glemte bogen og vendte ansigtet mod solen. Sådan sad hun en tid med lukkede øjne. Nede fra gaden lød der musik. Ciao, Anita bella. Hun vågnede ved lyden af hans stemme og førte en hånd op for at skygge for øjnene. En mand var trådt ud på tag-terrassen. Han kom langsomt gående over mod hende med en pose i hånden. Valzani stod der udenpå. Han strøg med hånden hen over hendes runde mave. Du bliver alt for tyk af al den chokolade, sagde han og holdt posen i vejret, så hun ikke kunne nå den. Jamen, den lille elsker chokolade. Jeg tror allerede, han kender ordet. Han sparkede, da du sagde det lige før. Thomas lagde sig på knæ og omfavnede hende. Chokolade, mumlede han med munden mod Anitas  mave. I det samme kom der et kraftigt stød af en lille bitte fod. Se selv. Thomas smilede og kær-tegnede hendes mave. Anita trak vejret i dybe drag. Hendes ånde-drag blev roligere og langsommere, jo tættere hun kom på fødslen. Hun var i niende måned nu og skulle føde sidst i maj. På Rigs-hospitalet hjemme i Danmark. Men barselsorloven skulle tilbringes her. Det var længe siden, de havde besluttet det. Stol på mig! havde Thomas sagt engang. Bare stol på mig. Det måtte være hundrede år siden. Mindst! Når man tænkte på alt, hvad der var sket siden. Og hun havde stolet på ham. Hun havde købt brude-kjolen, da det hele så værst ud. Alligevel havde hun sagt nej, da han friede til hende. Dengang havde hun ikke vidst, at hun var gravid. Hun havde været så vred på ham, at hun ikke syntes, hun kunne rumme al den vrede. Men da han stod uden for hendes hus et par dage senere, hvid i ansigtet og med jet-lag, havde hun lukket ham ind. Han fortalte historien, som den var. Uden at pynte på den, uden at lægge noget til eller trække fra. Da han havde tabt en del af hendes penge på køb af aktier, var han taget til Las Vegas som en sidste udvej. Han var taget over for at spille, som han før havde gjort. Og han havde gamblet, fortalte han. Selv om han havde lovet sig selv at lægge al den slags på hylden. I begyndelsen havde han tabt penge. Mange. Men så var der opstået en ny situation. Ved at holde på rødt hele vejen, endte jeg med at vinde pengene tilbage, fortalte han. Ja, og flere til! Anita nikkede tavs. Hun havde set, hvad der stod på kontoen i Rom og lod ham tale færdig. Nu spurgte hun: Hvad ville du have gjort, hvis du havde tabt resten af pengene? Selvfølgelig vandt jeg, sagde han. Ellers havde jeg fundet andre udveje. Han var ikke bange, ham Thomas. Han havde satset det hele - og vundet. Så kunne andre for hendes skyld mene om ham, hvad de ville. Vinden bar en duft af roser med sig fra busken midt på terrassen. Anita nulrede ham i håret. Sådan havde Lones brude-buket duftet. Den, som hun havde grebet. Ifølge over-troen ville hun blive den næste, der blev  gift. Og det var sket. Måske var det bestemt, allerede inden hun kom til brylluppet. Måske var alting bestemt i forvejen, tænkte hun. Også at hun ville komme tilbage til Rom. Sådan som spå-kvinden havde sagt. Thomas skiftede knæ. Så rejste han sig op med besvær. Jeg arme mand, sukkede han. Jeg kommer til at slæbe rundt på en kæmpe-kvinde fra Italien til Danmark og tilbage igen. Og alt sammen bare, fordi man kom til at love at samle en seng engang. Anita lo. Og lykkedes det så? Fik du nogen sinde samlet den? Shh! Sig dog ikke sådan noget. Man kunne let få ondt i ørerne. Jeg er ingen handy-mand. Jeg påtog mig kun at samle din seng, fordi jeg var vild med dig. Jeg ville have dig, og det er den skin-barlige sandhed. Jeg tror nu, det hele var forud-bestemt, sagde Anita drillende. Tror du det? Ja. Til brylluppet så Susanne et omvendt syvtal hænge over mit hoved. Det betød, at jeg ville komme til penge, sagde hun. Og det var jo netop lige sket. Da jeg så kom her til Rom, blev jeg spået, at jeg ville vende tilbage. Det må man jo også sige er sket. Vil det sige, at jeg slet ikke havde behøvet at slås med den seng? Det ser sådan ud. Nå, skidt. Hvad betyder en smule besvær? Og en tur fra Ikea til Las Vegas. Det var det hele værd, sagde han og kyssede hende. Kærligheden har mange navne, sagde han lidt efter. Men jeg kender kun det ene: Anita. Nede fra gaden lød musikken igen. Denne gang var det tonerne fra en trompet. Anita lyttede. Det lød fuldstændig som hendes far, der spillede."

mord_efter_mord = ["kapitel elve Det blev en trist dag. Alle følte sig nede. Den gale morder var ikke inden for rækkevidde længere. " ,
                  "Han var hoppet af krogen. Eller også havde han aldrig været på den. Iben sukkede. " ,
                  "Hvordan skulle de få fanget ham? Asger kørte hende hjem. Han var helt grå i huden. Det så ud, som om han havde tabt flere kilo. " ,
                  "Skal jeg gå med op? spurgte han. Han tændte en cigaret, idet hun åbnede døren. Nej, det behøver du ikke. " ,
                  "Du må hellere få hvilet dig lidt. Okay. Jeg orker heller ikke flere trapper i dag. Vi ses. Hun låste sig ind i lejligheden. " ,
                  "Hun var dødtræt. Det var alle. Derfor havde hun aftalt med Leo, at hun blev hjemme og ikke løb nogen steder. "]
                  # "Der var ingen til at passe på hende. Hun lavede et stykke kalkun-bryst med salat. Kødet blev enormt tørt. Kok blev hun aldrig. Hun smed det ud og gik i bad. Hun stod og tørrede sig foran spejlet. Hun kunne dårligt holde øjnene åbne. Det varme bad havde virkelig fået hende til at slappe af. Hun så på spejlet, som duggede. Pludselig mærkede hun pludselig en isnende kulde. Den lagde sig tæt om hende. Gåsehuden fik hendes hud til at krympe sig i varmen. I duggen trådte der bogstaver frem. Og hun læste: Bare vent. Det bliver dig inden længe. Han havde stået her i hendes badeværelse. Hvor var han nu? Aldrig havde hun følt sig så nøgen og sårbar. Hun tog undertøj på i en fart og løb ud i stuen. Hvor var hendes mobil? Hvor fanden havde hun smidt den? I stedet for at lede råbte hun til det skjulte kamera: Han er her! Eller har været her! Kom, for fanden!"

sort_sorg = ["I har været slemme! Sådan nogle modbydelige, slemme piger!  Stemmen var tyk og utydelig af følelse. " ,
            "Bente græd med en underlig høj, hikkende lyd. Susie knugede hende ind til sig og strøg hende over håret. " ,
            "Igen og igen, mens hun stirrede ud i mørket. De havde siddet alene i det lille kælder-rum i flere timer. " ,
            "Der var lige plads til dem i krogen mellem væggen og den store balje af plastic. Lige i begyndelsen troede de, at det var for sjov. " ,
            "Men ikke længere.  Rejs jer!  Den vrede stemme kom tilbage. Hårde, rasende hænder flåede deres tøj af, til de stod nøgne. " ,
            "Susie krydsede sine tynde arme over brystet. Hænderne tvang dem i små dragter. Kulden gjorde ondt langt ind i knoglerne.  "]
            # "Undskyld!  græd Bente.  Undskyld, hvis vi har gjort noget forkert! Det var ikke med vilje!  Sådan havde hun råbt mange gange, siden de blev puffet herned. Susie sagde ingenting. Hun vidste, at det ikke nyttede. Hun havde set ind i de ophidsede øjne. Susie havde kun én tanke i hovedet, mens hun stirrede ud i mørket: Så kom vi alligevel for sent hjem!  Dig først!  stønnede stemmen. Bente skreg, da de hårde hænder løftede hende op."


# trans_list = [anitas_millioner, mareridt, mord_efter_mord, sort_sorg]
# trans_list = [mord_efter_mord, sort_sorg]
trans_list = anitas_millioner + mareridt + mord_efter_mord + sort_sorg
# trans_list = sort_sorg

audio_path = "/home/karl/Desktop/random_search_data"
# audio_path = "/home/karl/Desktop/train/"
audio_dir = sorted(os.listdir(audio_path))

stats = np.array([])
for t in trans_list:
    stats = np.append(stats, len(t.split()))

print("mean: ", np.mean(stats))
print("std: ", np.std(stats))

def objective_function(alpha, beta, beam_width, audio_path, audio_dir, trans_list, prt = True):
    model = DanSpeechPrimary()
    recognizer = Recognizer(model=model)

    try:
        lm = DSL3gram()
        recognizer.update_decoder(lm=lm, alpha=alpha, beta=beta, beam_width=beam_width)
    except ImportError:
        print("ctcdecode not installed. Using greedy decoding.")

    results = 0
    weights = 0
    t = ""

    for audio, true_transcript in zip(audio_dir, trans_list):

        true_transcript = true_transcript.lower()
        true_transcript = true_transcript.replace(".", "")
        true_transcript = true_transcript.replace("é", "e")
        true_transcript = true_transcript.replace("\\", " ")
        true_transcript = true_transcript.replace(",", "")
        true_transcript = true_transcript.replace("?", "")
        true_transcript = true_transcript.replace("!", "")
        true_transcript = regex.sub(' +', ' ', true_transcript)
        true_transcript = true_transcript[:-1]


        # sound = AudioSegment.from_mp3(os.path.join(audio_path, audio))
        # sound.export("tmp.wav", format="wav")

        x, _ = librosa.load(os.path.join(audio_path, audio), sr=16000)
        sf.write("tmp.wav", x, 16000)
        audio = load_audio(path="tmp.wav")

        danspeech_transcript = recognizer.recognize(audio)
        if prt == True:
            print(len(true_transcript.split()))
            print(true_transcript)
            print(danspeech_transcript)
            print(jiwer.wer(true_transcript, danspeech_transcript))
            print("---------------------")
        weights += len(true_transcript.split())
        t += danspeech_transcript
        results += jiwer.wer(true_transcript, danspeech_transcript)*len(true_transcript.split())

    if prt == True:
        print("*******")
        print(results/weights, weights)
        print(t)

    return results/weights

# objective_function(0.65, 0.65, 227, audio_path, audio_dir, trans_list, prt = True)

results_best = 1
alpha_best = 1
beta_best = 1
beam_width_best = 1
exception_count = 0
count = 0

while True:
    alpha = np.random.permutation(np.arange(1,41)*0.05)[1]
    beta = np.random.permutation(np.arange(1,21)*0.1)[1]
    beam_width = np.random.permutation(np.arange(100,240))[1]

    try:
        count += 1
        print("**************************")
        print("Iteration number: ", count)
        print("**************************")

        results = objective_function(alpha, beta, beam_width, audio_path, audio_dir, trans_list, prt = False)
        print("RESULTS: ", results)
        if results < results_best:
            results_best = results
            alpha_best = alpha
            beta_best = beta
            beam_width_best = beam_width

            file = open("danspeechgridsearch_part2.txt", "a+")

            file.write("WER: " + str(round(results_best,3)) + "\n")
            file.write("alpha: " + str(round(alpha_best,2)) + "\n")
            file.write("beta: " + str(round(beta_best,2)) + "\n")
            file.write("beam width: " + str(beam_width_best) + "\n")
            file.write("-----------------------------------------\n")

            file.close()

    except Exception:
        exception_count += 1
        print("exception occured")
        print(exception_count)

    print("current try: alpha, beta, beam width: ", round(alpha,2), round(beta,2), beam_width)
    print("CURRENT BEST: results, alpha, beta, beam width: ", round(results_best,3),
          round(alpha_best,2), round(beta_best,2), beam_width_best)


