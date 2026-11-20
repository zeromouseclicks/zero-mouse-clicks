/* Zero Mouse Clicks — animated how-to for AI chats (beta).
   [data-howto]     the whole flow, played as a short film: ask → the chat finds the plan → open the link → PDF
   [data-connect]   adding the connector in Claude or ChatGPT, step by step
   [data-promptkit] a prompt you fill in, copy, open in ChatGPT or Claude, or share from the phone
   pre[data-copy] and .doc pre get a small copy button.
   No dependencies. Respects prefers-reduced-motion. */
(function () {
  'use strict';
  var MCP = 'https://ebqxcaelgmxofwtofrdc.supabase.co/functions/v1/zmc-ai/mcp';
  var SHEETS = ['G-001', 'A-102', 'A-101', 'A-103', 'A-104', 'A-105'];   // the floor plan sits on top, near the middle
  var RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  var T = {
    en: {
      tabs: 'Assistant', pause: 'Pause', play: 'Play',
      ask: 'Make me the plans of the 2-bedroom apartment in Rivera del Valle, Bogotá',
      search: 'Searching the web', find: 'Zero Mouse Clicks · find_floor_plans', found: '3 floor plans · 56 m²',
      link: 'Zero Mouse Clicks · plan_set_link', answer: 'Here is your plan set. Open the link:',
      pdf: 'plan-set.pdf · 6 A3 sheets', reading: 'Reading the plan…',
      steps: ['Ask for the plans', 'The chat finds the floor plan', 'Open the link', 'Your PDF, in a minute'],
      caps: ['Write it like you would to a person: project, city, apartment.', 'With the connector, the chat finds the project page, the floor plan and its area.',
             'One link. It opens zeromouseclicks.com with the plan already loaded.', 'The plan set is drawn in your browser: free, nothing is uploaded.'],
      cSteps: { claude: ['Settings › Connectors', 'Add custom connector', 'Name and this address', 'Done: use it in any chat'],
                chatgpt: ['Settings › Apps › Advanced: Developer mode', 'Create app', 'Name, address, no authentication', 'Done: add it to the chat with +'] },
      cUi: { settings: 'Settings', general: 'General', connectors: 'Connectors', apps: 'Apps', privacy: 'Privacy', add: 'Add custom connector',
             adv: 'Advanced settings', dev: 'Developer mode', create: 'Create app', name: 'Name', url: 'Remote MCP server URL',
             url2: 'MCP server URL', auth: 'Authentication', noauth: 'No authentication', ok: 'Add', ok2: 'Create',
             trust: 'I understand and want to continue', connected: 'Connected', newer: 'In some versions: Customize › Connectors, or Plugins › Add.' },
      copy: 'Copy', copied: 'Copied', share: 'Share…', openG: 'Open in ChatGPT', openC: 'Open in Claude',
      kFields: [['proj', 'Project', 'Rivera del Valle'], ['dev', 'Developer', 'Amarilo'], ['city', 'City', 'Bogotá'], ['apt', 'Apartment (optional)', '2 bedrooms, 56 m²']],
      kPh: { proj: '[project]', city: '[city]' },
      place: function (v, V) { return [["the apartment", 0]].concat(v.apt ? [[' (', 0], V('apt'), [')', 0]] : [])
        .concat([[' in the ', 0], V('proj'), [' project', 0]]).concat(v.dev ? [[' by ', 0], V('dev')] : []).concat([[', ', 0], V('city')]); },
      pYes: function (pl) { return [['Use Zero Mouse Clicks to make the plan set of ', 0]].concat(pl, [['. If there are several apartment types, do not ask me: give me one link per type.', 0]]); },
      pNo: function (pl) { return [['Search the web for the floor plan of ', 0]].concat(pl, [['. I need the direct address of the apartment floor plan image (a .jpg, .png or .webp file; not a web page and not the site plan of the complex). It is usually on the developer\'s project page or on listing portals, with words like plan, floor plan or type in the file name. If there are several apartment types, do not ask me: give me one link per type. Build each link like this:\nhttps://zeromouseclicks.com/?plan=IMAGE_URL&area=PRIVATE_AREA&unit=m2\nIf you cannot see the image address, give me the link of the page where the floor plan appears.', 0]]); },
      kWith: 'I added the connector', kWithout: 'No connector',
      kNote: 'On a phone, the buttons open the app if you have it. Or copy and paste the prompt in any AI chat.',
    },
    es: {
      tabs: 'Asistente', pause: 'Pausar', play: 'Reproducir',
      ask: 'Hazme los planos del apartamento de 2 alcobas en Rivera del Valle, Bogotá',
      search: 'Buscando en la web', find: 'Zero Mouse Clicks · find_floor_plans', found: '3 plantas · 56 m²',
      link: 'Zero Mouse Clicks · plan_set_link', answer: 'Aquí está tu juego de planos. Abre el enlace:',
      pdf: 'plan-set.pdf · 6 láminas A3', reading: 'Leyendo la planta…',
      steps: ['Pide los planos', 'El chat encuentra la planta', 'Abre el enlace', 'Tu PDF, en un minuto'],
      caps: ['Escríbelo como se lo dirías a una persona: proyecto, ciudad, apartamento.', 'Con el conector, el chat encuentra la página del proyecto, la planta y su área.',
             'Un enlace. Abre zeromouseclicks.com con la planta ya cargada.', 'El juego de planos se dibuja en tu navegador: gratis y sin subir nada.'],
      cSteps: { claude: ['Configuración › Conectores', 'Agregar conector personalizado', 'Nombre y esta dirección', 'Listo: úsalo en cualquier chat'],
                chatgpt: ['Configuración › Apps › Avanzado: modo desarrollador', 'Crear app', 'Nombre, dirección, sin autenticación', 'Listo: agrégala al chat con +'] },
      cUi: { settings: 'Configuración', general: 'General', connectors: 'Conectores', apps: 'Apps', privacy: 'Privacidad', add: 'Agregar conector personalizado',
             adv: 'Configuración avanzada', dev: 'Modo desarrollador', create: 'Crear app', name: 'Nombre', url: 'URL del servidor MCP remoto',
             url2: 'URL del servidor MCP', auth: 'Autenticación', noauth: 'Sin autenticación', ok: 'Agregar', ok2: 'Crear',
             trust: 'Entiendo y quiero continuar', connected: 'Conectado', newer: 'En algunas versiones: Personalizar › Conectores, o Plugins › Agregar.' },
      copy: 'Copiar', copied: 'Copiado', share: 'Compartir…', openG: 'Abrir en ChatGPT', openC: 'Abrir en Claude',
      kFields: [['proj', 'Proyecto', 'Rivera del Valle'], ['dev', 'Constructora', 'Amarilo'], ['city', 'Ciudad', 'Bogotá'], ['apt', 'Apartamento (opcional)', '2 alcobas, 56 m²']],
      kPh: { proj: '[proyecto]', city: '[ciudad]' },
      place: function (v, V) { return [['apartamento', 0]].concat(v.apt ? [[' (', 0], V('apt'), [')', 0]] : [])
        .concat([[' del proyecto ', 0], V('proj')]).concat(v.dev ? [[' de la constructora ', 0], V('dev')] : []).concat([[', en ', 0], V('city')]); },
      pYes: function (pl) { return [['Usa Zero Mouse Clicks para hacer el juego de planos del ', 0]].concat(pl, [['. Si hay varias tipologías, no me preguntes: dame un enlace por cada una.', 0]]); },
      pNo: function (pl) { return [['Busca en la web la planta del ', 0]].concat(pl, [['. Necesito la dirección directa de la imagen de la planta del apartamento (un archivo .jpg, .png o .webp; no una página ni la implantación del conjunto). Suele estar en la página del proyecto de la constructora o en portales inmobiliarios como Ciencuadras o Fincaraíz, con palabras como plano, planta o tipo en el nombre del archivo. Si hay varias tipologías, no me preguntes: dame un enlace por cada una. Arma cada enlace así:\nhttps://zeromouseclicks.com/?plan=URL_DE_LA_IMAGEN&area=AREA_PRIVADA&unit=m2\nSi no logras ver la dirección de la imagen, dame el enlace de la página donde aparece la planta.', 0]]); },
      kWith: 'Ya agregué el conector', kWithout: 'Sin conector',
      kNote: 'En el celular, los botones abren la app si la tienes. O copia el texto y pégalo en cualquier chat de IA.',
    },
    fr: {
      tabs: 'Assistant', pause: 'Pause', play: 'Lecture',
      ask: "Fais-moi les plans de l'appartement 2 chambres de Rivera del Valle, à Bogotá",
      search: 'Recherche sur le Web', find: 'Zero Mouse Clicks · find_floor_plans', found: '3 plans · 56 m²',
      link: 'Zero Mouse Clicks · plan_set_link', answer: 'Voici votre jeu de plans. Ouvrez le lien :',
      pdf: 'plan-set.pdf · 6 feuilles A3', reading: 'Lecture du plan…',
      steps: ['Demandez les plans', "L'assistant trouve le plan", 'Ouvrez le lien', 'Votre PDF, en une minute'],
      caps: ['Écrivez comme à une personne : projet, ville, appartement.', "Avec le connecteur, l'assistant trouve la page du projet, le plan et sa superficie.",
             'Un seul lien. Il ouvre zeromouseclicks.com avec le plan déjà chargé.', 'Le jeu de plans est dessiné dans votre navigateur : gratuit, rien n’est téléversé.'],
      cSteps: { claude: ['Paramètres › Connecteurs', 'Ajouter un connecteur personnalisé', 'Un nom et cette adresse', 'C’est fait : utilisez-le dans tout chat'],
                chatgpt: ['Paramètres › Apps › Avancé : mode développeur', 'Créer une app', 'Nom, adresse, sans authentification', 'C’est fait : ajoutez-la au chat avec +'] },
      cUi: { settings: 'Paramètres', general: 'Général', connectors: 'Connecteurs', apps: 'Apps', privacy: 'Confidentialité', add: 'Ajouter un connecteur personnalisé',
             adv: 'Paramètres avancés', dev: 'Mode développeur', create: 'Créer une app', name: 'Nom', url: 'URL du serveur MCP distant',
             url2: 'URL du serveur MCP', auth: 'Authentification', noauth: 'Aucune authentification', ok: 'Ajouter', ok2: 'Créer',
             trust: 'Je comprends et je veux continuer', connected: 'Connecté', newer: 'Selon la version : Personnaliser › Connecteurs, ou Plugins › Ajouter.' },
      copy: 'Copier', copied: 'Copié', share: 'Partager…', openG: 'Ouvrir dans ChatGPT', openC: 'Ouvrir dans Claude',
      kFields: [['proj', 'Projet', 'Rivera del Valle'], ['dev', 'Promoteur', 'Amarilo'], ['city', 'Ville', 'Bogotá'], ['apt', 'Appartement (facultatif)', '2 chambres, 56 m²']],
      kPh: { proj: '[projet]', city: '[ville]' },
      place: function (v, V) { return [["l'appartement", 0]].concat(v.apt ? [[' (', 0], V('apt'), [')', 0]] : [])
        .concat([[' du projet ', 0], V('proj')]).concat(v.dev ? [[' du promoteur ', 0], V('dev')] : []).concat([[', à ', 0], V('city')]); },
      pYes: function (pl) { return [['Utilise Zero Mouse Clicks pour faire le jeu de plans de ', 0]].concat(pl, [[". S'il y a plusieurs types d'appartements, ne me pose pas de question : donne-moi un lien pour chacun.", 0]]); },
      pNo: function (pl) { return [['Cherche sur le Web le plan de ', 0]].concat(pl, [[". J'ai besoin de l'adresse directe de l'image du plan de l'appartement (un fichier .jpg, .png ou .webp; pas une page ni le plan d'implantation de l'ensemble). Elle se trouve souvent sur la page du projet du promoteur ou sur des portails immobiliers, avec des mots comme plan, type ou unité dans le nom du fichier. S'il y a plusieurs types d'appartements, ne me pose pas de question : donne-moi un lien pour chacun. Construis chaque lien ainsi :\nhttps://zeromouseclicks.com/?plan=URL_IMAGE&area=SUPERFICIE_PRIVATIVE&unit=m2\nSi tu ne vois pas l'adresse de l'image, donne-moi le lien de la page où se trouve le plan.", 0]]); },
      kWith: "J'ai ajouté le connecteur", kWithout: 'Sans connecteur',
      kNote: "Sur un téléphone, les boutons ouvrent l'application si vous l'avez. Sinon, copiez le texte et collez-le dans n'importe quel assistant IA.",
    }
  };

  /* ---------------------------------------------------------------- style */
  var CSS = [
    '.zh,.zc,.zk{--zl:#262624;--zf:#f5f5f3;--zm:#85857f;--zi:#9fd0ff;--zg:#a0ffc8;--zb:#0b0b0a;font-family:inherit;color:var(--zf)}',
    '.zh{margin:28px 0 8px}',
    '.zh-top,.zc-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}',
    '.zt{display:inline-flex;gap:2px;border:1px solid var(--zl);border-radius:999px;padding:3px}',
    '.zt button{font-weight:500;font-size:13px;line-height:1;font-family:inherit;color:var(--zm);background:none;border:0;border-radius:999px;padding:7px 14px;cursor:pointer}',
    '.zt button[aria-selected=true]{background:#1c1c1a;color:var(--zf)}',
    '.zpp{width:32px;height:32px;border-radius:50%;border:1px solid var(--zl);background:none;color:var(--zm);cursor:pointer;font-size:11px}',
    '.zpp:hover{color:var(--zf)}',
    '.zw{position:relative;border:1px solid var(--zl);border-radius:16px;background:var(--zb);overflow:hidden;height:360px;box-shadow:0 30px 80px rgba(0,0,0,.5)}',
    '.zw-bar{display:flex;align-items:center;gap:6px;height:36px;padding:0 14px;border-bottom:1px solid var(--zl);font:12px/1 "JetBrains Mono",ui-monospace,monospace;color:var(--zm)}',
    '.zw-bar i{width:8px;height:8px;border-radius:50%;background:#2a2a28}.zw-bar span{margin-left:8px}',
    '.zh-msgs{position:absolute;inset:36px 0 64px;padding:18px 18px 0;display:flex;flex-direction:column;gap:10px;overflow:hidden}',
    '.zh-u{align-self:flex-end;max-width:82%;background:#1d1d1b;border-radius:16px 16px 4px 16px;padding:10px 14px;font-size:14px;line-height:1.4}',
    '.zh-a{align-self:flex-start;max-width:90%;font-size:14px;line-height:1.45;color:#dcdcd6}',
    '.zh-chip{display:flex;align-items:center;gap:8px;font:12px/1.3 "JetBrains Mono",ui-monospace,monospace;color:var(--zm);padding:6px 10px;border:1px solid var(--zl);border-radius:10px;width:max-content;max-width:100%}',
    '.zh-chip s{text-decoration:none;color:var(--zg)}.zh-chip em{font-style:normal;color:var(--zf)}',
    '.zspin{width:10px;height:10px;border-radius:50%;border:2px solid #3a3a37;border-top-color:var(--zi);animation:zspin .7s linear infinite;flex:none}',
    '.zok{width:10px;height:10px;border-radius:50%;background:var(--zg);box-shadow:0 0 8px var(--zg);flex:none}',
    '@keyframes zspin{to{transform:rotate(360deg)}}',
    '.zh-link{display:inline-block;margin-top:8px;padding:8px 12px;border-radius:10px;background:rgba(159,208,255,.1);color:var(--zi);font:12.5px/1.3 "JetBrains Mono",ui-monospace,monospace;border:1px solid rgba(159,208,255,.25)}',
    '.zh-link.hit{background:rgba(159,208,255,.25)}',
    '.zh-in{position:absolute;left:14px;right:14px;bottom:14px;height:40px;border:1px solid var(--zl);border-radius:12px;display:flex;align-items:center;padding:0 46px 0 14px;font-size:14px;color:var(--zf);white-space:nowrap;overflow:hidden}',
    '.zh-in .ph{color:#55554f}.zcar{display:inline-block;width:1.5px;height:17px;background:var(--zf);margin-left:1px;vertical-align:-3px;animation:zblink 1s steps(1) infinite}',
    '@keyframes zblink{50%{opacity:0}}',
    '.zh-send{position:absolute;right:6px;top:6px;width:28px;height:28px;border-radius:8px;background:#2a2a28;color:#000;display:grid;place-items:center;font-weight:700;font-size:14px;transition:background .2s}',
    '.zh-send.on{background:var(--zf)}',
    '.zh-site{position:absolute;inset:0;background:#000;transform:translateY(102%);transition:transform .55s cubic-bezier(.2,.8,.2,1);display:flex;flex-direction:column}',
    '.zh-site.on{transform:none}',
    '.zh-url{margin:9px 12px;height:22px;border-radius:7px;background:#151514;font:11.5px/22px "JetBrains Mono",ui-monospace,monospace;color:var(--zm);padding:0 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '.zh-url b{color:var(--zf);font-weight:500}',
    '.zh-stage{position:relative;flex:1;margin:0 12px 12px;border-radius:10px;overflow:hidden;background:#0d0d0c}',
    '.zh-plan{position:absolute;left:50%;top:50%;height:86%;transform:translate(-50%,-50%);border-radius:6px;transition:opacity .5s,transform .6s}',
    '.zh-scan{position:absolute;left:0;right:0;height:2px;background:var(--zi);box-shadow:0 0 18px var(--zi);top:6%;opacity:0}',
    '.zh-scan.go{opacity:1;animation:zscan 1.5s ease-in-out forwards}',
    '@keyframes zscan{from{top:7%}to{top:93%}}',
    '.zh-read{position:absolute;left:12px;top:10px;font:11px/1 "JetBrains Mono",ui-monospace,monospace;color:var(--zi)}',
    '.zh-fan img{position:absolute;left:50%;top:46%;width:44%;border-radius:4px;box-shadow:0 10px 30px rgba(0,0,0,.6);opacity:0;transform:translate(-50%,-50%) scale(.7);transition:transform .7s cubic-bezier(.2,.8,.2,1),opacity .4s}',
    '.zh-pdf{position:absolute;left:50%;bottom:12px;transform:translate(-50%,10px);opacity:0;transition:all .4s;background:var(--zf);color:#000;border-radius:999px;padding:7px 14px;font-weight:600;font-size:12.5px;line-height:1;font-family:inherit;white-space:nowrap}',
    '.zh-pdf.on{opacity:1;transform:translate(-50%,0)}',
    '.zcur{position:absolute;width:18px;height:18px;left:60%;top:110%;transition:left .8s cubic-bezier(.3,.7,.2,1),top .8s cubic-bezier(.3,.7,.2,1);pointer-events:none;z-index:5}',
    '.zcur svg{width:18px;height:18px;filter:drop-shadow(0 1px 2px rgba(0,0,0,.7))}',
    '.zcur.click::after{content:"";position:absolute;left:-10px;top:-10px;width:22px;height:22px;border-radius:50%;border:2px solid var(--zi);animation:zripple .5s ease-out forwards}',
    '@keyframes zripple{from{transform:scale(.3);opacity:1}to{transform:scale(1.6);opacity:0}}',
    '.zs{list-style:none;display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0 0!important;padding:0}',
    '.zs li{margin:0!important;cursor:pointer;font-size:13px;line-height:1.3;color:var(--zm);padding:0}',
    '.zs li b{display:block;font:500 11px/1 "JetBrains Mono",ui-monospace,monospace;margin-bottom:6px;color:inherit}',
    '.zs li i{display:block;height:2px;background:var(--zl);border-radius:2px;margin-bottom:8px;overflow:hidden}',
    '.zs li i::after{content:"";display:block;height:100%;width:0;background:var(--zi)}',
    '.zs li.done{color:#b9b9b3}.zs li.done i::after{width:100%}',
    '.zs li.cur{color:var(--zf)}.zs li.cur i::after{animation:zfill var(--d,3s) linear forwards}',
    '.zs.paused li.cur i::after{animation-play-state:paused}',
    '@keyframes zfill{to{width:100%}}',
    '.zcap{margin-top:10px!important;font-size:14px!important;color:#b9b9b3!important;min-height:2.6em}',
    '@media (max-width:620px){.zw{height:330px}.zs{grid-template-columns:repeat(2,1fr)}.zh-u,.zh-a{font-size:13px}}',
    /* connector film */
    '.zc{margin:22px 0 6px}',
    '.zc .zw{height:340px;display:flex}',
    '.zc-side{width:30%;max-width:170px;border-right:1px solid var(--zl);padding:46px 8px 8px;display:flex;flex-direction:column;gap:2px}',
    '.zc-side span{font-size:12.5px;color:var(--zm);padding:7px 10px;border-radius:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:all .3s}',
    '.zc-side span.on{background:#1c1c1a;color:var(--zf)}',
    '.zc-main{position:relative;flex:1;padding:46px 18px 14px;overflow:hidden}',
    '.zc-h{font-weight:600;font-size:15px;margin-bottom:12px}',
    '.zc-row{display:flex;align-items:center;justify-content:space-between;gap:10px;border:1px solid var(--zl);border-radius:10px;padding:10px 12px;font-size:13px;color:#cfcfc9;margin-bottom:8px;transition:all .3s}',
    '.zc-row .ok{color:var(--zg);font:12px/1 "JetBrains Mono",ui-monospace,monospace}',
    '.zc-btn{display:inline-block;border:1px solid var(--zl);border-radius:999px;padding:7px 12px;font-size:12.5px;color:var(--zf);transition:all .25s}',
    '.zc-btn.hot{background:var(--zf);color:#000;border-color:var(--zf)}',
    '.ztog{width:30px;height:17px;border-radius:999px;background:#2a2a28;position:relative;transition:background .3s;flex:none}',
    '.ztog::after{content:"";position:absolute;left:2px;top:2px;width:13px;height:13px;border-radius:50%;background:#77776f;transition:all .3s}',
    '.ztog.on{background:var(--zg)}.ztog.on::after{left:15px;background:#000}',
    '.zc-modal{position:absolute;inset:38px 14px 12px;background:#121211;border:1px solid #333330;border-radius:12px;padding:14px;transform:translateY(16px);opacity:0;transition:all .35s;pointer-events:none}',
    '.zc-modal.on{transform:none;opacity:1}',
    '.zc-f{margin-bottom:8px}.zc-f label{display:block;font-size:11px;color:var(--zm);margin-bottom:4px}',
    '.zc-f div{height:28px;border:1px solid var(--zl);border-radius:8px;padding:0 9px;font:12px/27px "JetBrains Mono",ui-monospace,monospace;color:var(--zf);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '.zc-f div.sel{font-family:inherit;font-size:12.5px}',
    '.zc-chk{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--zm);margin:4px 0 10px}',
    '.zc-chk i{width:12px;height:12px;border:1px solid #55554f;border-radius:3px;display:inline-block}.zc-chk i.on{background:var(--zg);border-color:var(--zg)}',
    '.zc-note{font-size:12.5px!important;color:var(--zm)!important;margin-top:8px!important}',
    '@media (max-width:620px){.zc .zw{height:350px}.zc-side{display:none}.zc-main{padding-top:46px}}',
    /* prompt kit */
    '.zk{margin:20px 0 8px;border:1px solid var(--zl);border-radius:16px;padding:18px;background:#070706}',
    '.zk-f{display:grid;grid-template-columns:1fr 1fr;gap:10px}',
    '.zk code .v{color:var(--zi)}.zk code .ph{color:#ffc878}',
    '.zk-f label{font-size:12px;color:var(--zm);display:flex;flex-direction:column;gap:6px}',
    '.zk-f input{font-size:15px;line-height:1.2;font-family:inherit;color:var(--zf);background:#000;border:1px solid var(--zl);border-radius:10px;padding:11px 12px;outline:none;width:100%}',
    '.zk-f input:focus{border-color:#4a4a46}',
    '.zk .zt{margin:14px 0 0}',
    '.zk-box{position:relative;margin-top:12px}',
    '.zk-box pre{margin:0!important;padding:16px 52px 16px 16px!important;white-space:pre-wrap;overflow-wrap:anywhere}',
    '.zk-go{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}',
    '.zk-go a,.zk-go button{display:inline-flex;align-items:center;gap:7px;font-weight:600;font-size:14px;line-height:1;font-family:inherit;text-decoration:none!important;border-radius:999px;padding:11px 16px;cursor:pointer;border:1px solid var(--zl);background:none;color:var(--zf)!important}',
    '.zk-go a.pri{background:var(--zf);color:#000!important;border-color:var(--zf)}',
    '.zk-go a:hover,.zk-go button:hover{box-shadow:0 0 0 5px rgba(245,245,243,.06)}',
    '.zk-note{font-size:13px!important;color:var(--zm)!important;margin-top:10px!important}',
    '@media (max-width:620px){.zk-f{grid-template-columns:1fr}.zk-go a,.zk-go button{flex:1 1 auto;justify-content:center}}',
    /* copy button */
    '.zcp-wrap{position:relative}',
    '.zcp{position:absolute;top:8px;right:8px;display:inline-flex;align-items:center;gap:6px;height:30px;min-width:30px;padding:0 8px;border-radius:8px;border:1px solid var(--zl,#262624);background:#141413;color:#b9b9b3;cursor:pointer;font:12px/1 "JetBrains Mono",ui-monospace,monospace;opacity:.85;transition:opacity .2s,color .2s}',
    '.zcp:hover,.zcp:focus-visible{opacity:1;color:#f5f5f3}',
    '.zcp svg{width:14px;height:14px}.zcp span{display:none}.zcp.ok{color:#a0ffc8}.zcp.ok span{display:inline}',
    '@media (hover:none){.zcp{opacity:1;height:36px;min-width:36px;justify-content:center}}'
  ].join('\n');
  function style() {
    if (document.getElementById('zh-css')) return;
    var s = document.createElement('style'); s.id = 'zh-css'; s.textContent = CSS; document.head.appendChild(s);
  }

  /* ---------------------------------------------------------------- helpers */
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function lang(node) {
    var l = (node.getAttribute('data-lang') || document.documentElement.lang || 'en').slice(0, 2);
    return T[l] ? l : 'en';
  }
  var ICON_COPY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg>';
  var ICON_OK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  var CURSOR = '<svg viewBox="0 0 24 24"><path d="M5 3l14 8-6 1.6L10 19z" fill="#fff" stroke="#000" stroke-width="1.4" stroke-linejoin="round"/></svg>';

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (res, rej) {
      var t = document.createElement('textarea'); t.value = text; t.setAttribute('readonly', '');
      t.style.position = 'fixed'; t.style.opacity = '0'; document.body.appendChild(t); t.select();
      try { document.execCommand('copy') ? res() : rej(); } catch (e) { rej(e); } document.body.removeChild(t);
    });
  }
  function copyButton(getText, L) {
    var b = el('button', 'zcp', ICON_COPY + '<span>' + esc(L.copied) + '</span>');
    b.type = 'button'; b.setAttribute('aria-label', L.copy); b.title = L.copy;
    b.addEventListener('click', function () {
      copyText(getText()).then(function () {
        b.classList.add('ok'); b.innerHTML = ICON_OK + '<span>' + esc(L.copied) + '</span>';
        clearTimeout(b._t); b._t = setTimeout(function () { b.classList.remove('ok'); b.innerHTML = ICON_COPY + '<span>' + esc(L.copied) + '</span>'; }, 1800);
      }, function () {});
    });
    return b;
  }
  /* a small runner: steps are async functions; a new run cancels the old one */
  function Runner() { this.id = 0; }
  Runner.prototype.wait = function (ms, id) { var self = this; return new Promise(function (res, rej) { setTimeout(function () { self.id === id ? res() : rej('stop'); }, ms); }); };
  function visible(node, cb) {
    if (!('IntersectionObserver' in window)) { cb(true); return; }
    new IntersectionObserver(function (es) { cb(es[0].isIntersecting); }, { threshold: 0.25 }).observe(node);
  }
  function tabs(names, cur, onPick, label) {
    var t = el('div', 'zt'); t.setAttribute('role', 'tablist'); t.setAttribute('aria-label', label);
    names.forEach(function (n) {
      var b = el('button', '', esc(n[1])); b.type = 'button'; b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', n[0] === cur ? 'true' : 'false');
      b.addEventListener('click', function () { [].forEach.call(t.children, function (c) { c.setAttribute('aria-selected', c === b ? 'true' : 'false'); }); onPick(n[0]); });
      t.appendChild(b);
    });
    return t;
  }
  function stepList(labels, onPick) {
    var ol = el('ol', 'zs');
    labels.forEach(function (s, i) {
      var li = el('li', '', '<i></i><b>' + (i + 1) + '</b>' + esc(s)); li.tabIndex = 0;
      li.addEventListener('click', function () { onPick(i); });
      li.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onPick(i); } });
      ol.appendChild(li);
    });
    return ol;
  }
  function markSteps(ol, i, dur) {
    [].forEach.call(ol.children, function (li, k) {
      li.className = k < i ? 'done' : k === i ? 'cur' : '';
      if (k === i) { li.style.setProperty('--d', (dur / 1000) + 's'); var b = li.querySelector('i'); b.replaceWith(b.cloneNode()); }
    });
  }

  /* ---------------------------------------------------------------- the whole flow */
  function howto(root) {
    var L = T[lang(root)], ai = root.getAttribute('data-ai') || 'chatgpt';
    var R = new Runner(), on = false, paused = false, step = 0;
    var DUR = [4300, 4600, 3000, 5200];
    root.classList.add('zh');
    var top = el('div', 'zh-top');
    var tb = tabs([['chatgpt', 'ChatGPT'], ['claude', 'Claude']], ai, function (a) { ai = a; name.textContent = a === 'claude' ? 'Claude' : 'ChatGPT'; play(0); }, L.tabs);
    var pp = el('button', 'zpp', '❚❚'); pp.type = 'button'; pp.setAttribute('aria-label', L.pause);
    top.appendChild(tb); top.appendChild(pp); root.appendChild(top);
    var win = el('div', 'zw'); win.setAttribute('aria-hidden', 'true');
    var bar = el('div', 'zw-bar', '<i></i><i></i><i></i><span></span>'); var name = bar.querySelector('span'); name.textContent = ai === 'claude' ? 'Claude' : 'ChatGPT';
    var msgs = el('div', 'zh-msgs');
    var inp = el('div', 'zh-in', '<span class="tx"></span><span class="zcar"></span><b class="zh-send">↑</b>');
    var tx = inp.querySelector('.tx'), send = inp.querySelector('.zh-send');
    var site = el('div', 'zh-site', '<div class="zh-url"><b>zeromouseclicks.com</b>/?plan=https://…/rivera-plano.jpg&amp;area=56</div><div class="zh-stage"><img class="zh-plan" alt=""><div class="zh-scan"></div><div class="zh-read"></div><div class="zh-fan"></div><div class="zh-pdf"></div></div>');
    var plan = site.querySelector('.zh-plan'), scan = site.querySelector('.zh-scan'), read = site.querySelector('.zh-read'), fan = site.querySelector('.zh-fan'), pdf = site.querySelector('.zh-pdf');
    pdf.textContent = '✓ ' + L.pdf;
    SHEETS.forEach(function () { fan.appendChild(el('img')); });
    var cur = el('div', 'zcur', CURSOR);
    win.appendChild(bar); win.appendChild(msgs); win.appendChild(inp); win.appendChild(site); win.appendChild(cur);
    root.appendChild(win);
    var ol = stepList(L.steps, function (i) { paused = false; pp.textContent = '❚❚'; play(i); }); root.appendChild(ol);
    var cap = el('p', 'zcap'); cap.setAttribute('aria-live', 'polite'); root.appendChild(cap);
    var loaded = false;
    function load() {
      if (loaded) return; loaded = true;
      plan.src = '/img/brochure.jpg';
      [].forEach.call(fan.children, function (im, i) { im.src = '/img/sheets/' + SHEETS[i] + '_s.jpg'; });
    }
    function chip(text, sub) {
      var c = el('div', 'zh-chip', '<span class="zspin"></span><em>' + esc(text) + '</em>' + (sub ? ' <s></s>' : ''));
      msgs.appendChild(c); return c;
    }
    function done(c, sub) { c.querySelector('.zspin').className = 'zok'; if (sub) c.querySelector('s').textContent = sub; }
    function reset() {
      msgs.innerHTML = ''; tx.innerHTML = '<span class="ph">…</span>'; send.classList.remove('on');
      site.classList.remove('on'); scan.classList.remove('go'); read.textContent = ''; pdf.classList.remove('on');
      plan.style.opacity = '1'; plan.style.transform = 'translate(-50%,-50%)';
      [].forEach.call(fan.children, function (im) { im.style.opacity = '0'; im.style.transform = 'translate(-50%,-50%) scale(.7)'; });
      cur.className = 'zcur'; cur.style.left = '60%'; cur.style.top = '110%';
    }
    // each step: (id, fast) → promise. fast = draw its final state at once
    var S = [
      function (id, fast) {
        tx.textContent = ''; var s = L.ask;
        if (fast) { tx.textContent = ''; msgs.appendChild(el('div', 'zh-u', esc(s))); return Promise.resolve(); }
        var i = 0;
        function tick() { if (R.id !== id) return Promise.reject('stop'); if (i >= s.length) return Promise.resolve();
          i += 2; tx.textContent = s.slice(0, i); send.classList.toggle('on', i > 0); return R.wait(38, id).then(tick); }
        return R.wait(300, id).then(tick).then(function () { return R.wait(350, id); }).then(function () {
          tx.innerHTML = '<span class="ph">…</span>'; send.classList.remove('on'); msgs.appendChild(el('div', 'zh-u', esc(s)));
        });
      },
      function (id, fast) {
        var a = chip(L.search), b, c;
        if (fast) { done(a); done(b = chip(L.find, 1), L.found); done(chip(L.link)); return Promise.resolve(); }
        return R.wait(1100, id).then(function () { done(a); b = chip(L.find, 1); return R.wait(1500, id); })
          .then(function () { done(b, L.found); c = chip(L.link); return R.wait(1100, id); }).then(function () { done(c); return R.wait(300, id); });
      },
      function (id, fast) {
        var m = el('div', 'zh-a', esc(L.answer) + '<br><span class="zh-link">zeromouseclicks.com/?plan=…&amp;area=56</span>');
        msgs.appendChild(m);
        if (fast) return Promise.resolve();
        var lk = m.querySelector('.zh-link');
        return R.wait(500, id).then(function () {
          var wr = win.getBoundingClientRect(), r = lk.getBoundingClientRect();
          cur.style.left = ((r.left - wr.left + r.width * 0.45) / wr.width * 100) + '%';
          cur.style.top = ((r.top - wr.top + r.height * 0.55) / wr.height * 100) + '%';
          return R.wait(1000, id);
        }).then(function () { cur.classList.add('click'); lk.classList.add('hit'); return R.wait(550, id); });
      },
      function (id, fast) {
        cur.className = 'zcur'; cur.style.top = '110%'; site.classList.add('on');
        function sheets() {
          plan.style.opacity = '.0'; plan.style.transform = 'translate(-50%,-50%) scale(.9)'; read.textContent = '';
          [].forEach.call(fan.children, function (im, i) {
            var k = i - (fan.children.length - 1) / 2;
            im.style.opacity = '1'; im.style.zIndex = String(10 - Math.abs(i - 2));
            im.style.transform = 'translate(calc(-50% + ' + (k * 15) + '%), calc(-50% + ' + (Math.abs(k) * 3) + '%)) rotate(' + (k * 4.5) + 'deg)';
          });
        }
        if (fast) { sheets(); pdf.classList.add('on'); return Promise.resolve(); }
        return R.wait(600, id).then(function () { read.textContent = L.reading; scan.classList.add('go'); return R.wait(1700, id); })
          .then(function () { scan.classList.remove('go'); sheets(); return R.wait(900, id); })
          .then(function () { pdf.classList.add('on'); return R.wait(2000, id); });
      }
    ];
    function play(from) {
      var id = ++R.id; step = from; load(); reset();
      for (var k = 0; k < from; k++) S[k](id, true);
      ol.classList.remove('paused');
      (function next(i) {
        if (i >= S.length) { return R.wait(400, id).then(function () { play(0); }, function () {}); }
        step = i; markSteps(ol, i, DUR[i]); cap.textContent = L.caps[i];
        return S[i](id, false).then(function () { return next(i + 1); }, function () {});
      })(from);
    }
    function stop() { R.id++; ol.classList.add('paused'); }
    pp.addEventListener('click', function () {
      paused = !paused; pp.textContent = paused ? '▶' : '❚❚'; pp.setAttribute('aria-label', paused ? L.play : L.pause);
      if (paused) stop(); else play(step);
    });
    if (RM) { load(); reset(); var id0 = ++R.id; S.forEach(function (f, i) { if (i < 3) f(id0, true); }); markSteps(ol, 4, 1); cap.textContent = L.caps[0]; pp.style.display = 'none'; return; }
    visible(root, function (v) { if (v && !on && !paused) { on = true; play(step); } else if (!v && on) { on = false; stop(); } });
  }

  /* ---------------------------------------------------------------- connector film */
  function connect(root) {
    var L = T[lang(root)], U = L.cUi, ai = root.getAttribute('data-ai') || 'claude', url = root.getAttribute('data-url') || MCP;
    var R = new Runner(), on = false, paused = false, step = 0;
    root.classList.add('zc');
    var top = el('div', 'zc-top');
    var tb = tabs([['claude', 'Claude'], ['chatgpt', 'ChatGPT']], ai, function (a) { ai = a; build(); play(0); }, L.tabs);
    var pp = el('button', 'zpp', '❚❚'); pp.type = 'button'; pp.setAttribute('aria-label', L.pause);
    top.appendChild(tb); top.appendChild(pp); root.appendChild(top);
    var win = el('div', 'zw'); win.setAttribute('aria-hidden', 'true');
    var side = el('div', 'zc-side'), main = el('div', 'zc-main');
    var bar = el('div', 'zw-bar', '<i></i><i></i><i></i><span></span>'); bar.style.cssText = 'position:absolute;left:0;right:0;top:0';
    var cur = el('div', 'zcur', CURSOR);
    win.appendChild(bar); win.appendChild(side); win.appendChild(main); win.appendChild(cur); root.appendChild(win);
    var ol = el('ol', 'zs'); root.appendChild(ol);
    var note = el('p', 'zc-note'); root.appendChild(note);
    var parts = {};
    function build() {
      var st = L.cSteps[ai];
      bar.querySelector('span').textContent = (ai === 'claude' ? 'Claude' : 'ChatGPT') + ' · ' + U.settings;
      side.innerHTML = ''; [U.general, ai === 'claude' ? U.connectors : U.apps, U.privacy].forEach(function (s, i) { var x = el('span', '', esc(s)); side.appendChild(x); parts['side' + i] = x; });
      var conn = ai === 'claude';
      main.innerHTML = '<div class="zc-h">' + esc(conn ? U.connectors : U.apps) + '</div>' +
        (conn ? '' : '<div class="zc-row" data-k="adv"><span>' + esc(U.adv) + ' · ' + esc(U.dev) + '</span><i class="ztog"></i></div>') +
        '<div class="zc-row" data-k="list" style="opacity:.0"><span>Zero Mouse Clicks</span><span class="ok">● ' + esc(U.connected) + '</span></div>' +
        '<span class="zc-btn" data-k="add">' + esc(conn ? U.add : U.create) + '</span>' +
        '<div class="zc-modal"><div class="zc-h">' + esc(conn ? U.add : U.create) + '</div>' +
        '<div class="zc-f"><label>' + esc(U.name) + '</label><div data-k="name"></div></div>' +
        '<div class="zc-f"><label>' + esc(conn ? U.url : U.url2) + '</label><div data-k="url"></div></div>' +
        (conn ? '' : '<div class="zc-f"><label>' + esc(U.auth) + '</label><div class="sel">' + esc(U.noauth) + '</div></div><div class="zc-chk"><i data-k="chk"></i>' + esc(U.trust) + '</div>') +
        '<span class="zc-btn" data-k="ok">' + esc(conn ? U.ok : U.ok2) + '</span></div>';
      ['adv', 'list', 'add', 'name', 'url', 'chk', 'ok'].forEach(function (k) { parts[k] = main.querySelector('[data-k="' + k + '"]'); });
      parts.modal = main.querySelector('.zc-modal'); parts.tog = main.querySelector('.ztog');
      ol.innerHTML = ''; st.forEach(function (s, i) {
        var li = el('li', '', '<i></i><b>' + (i + 1) + '</b>' + esc(s)); li.tabIndex = 0;
        li.addEventListener('click', function () { paused = false; pp.textContent = '❚❚'; play(i); }); ol.appendChild(li);
      });
      note.textContent = conn ? U.newer : '';
    }
    function point(node, id) {
      var wr = win.getBoundingClientRect(), r = node.getBoundingClientRect();
      cur.style.left = ((r.left - wr.left + Math.min(r.width * 0.5, 60)) / wr.width * 100) + '%';
      cur.style.top = ((r.top - wr.top + r.height * 0.6) / wr.height * 100) + '%';
      return R.wait(850, id).then(function () { cur.classList.remove('click'); void cur.offsetWidth; cur.classList.add('click'); return R.wait(380, id); });
    }
    function type(node, text, id, fast) {
      if (fast) { node.textContent = text; return Promise.resolve(); }
      var i = 0; node.textContent = '';
      function tick() { if (R.id !== id) return Promise.reject('stop'); if (i >= text.length) return Promise.resolve(); i += 3; node.textContent = text.slice(0, i); return R.wait(24, id).then(tick); }
      return tick();
    }
    function reset() {
      [0, 1, 2].forEach(function (i) { parts['side' + i].classList.remove('on'); });
      parts.list.style.opacity = '0'; parts.add.classList.remove('hot'); parts.modal.classList.remove('on');
      parts.name.textContent = ''; parts.url.textContent = ''; parts.ok.classList.remove('hot');
      if (parts.tog) parts.tog.classList.remove('on'); if (parts.chk) parts.chk.classList.remove('on');
      cur.className = 'zcur'; cur.style.left = '50%'; cur.style.top = '110%';
    }
    var S = [
      function (id, fast) {
        if (fast) { parts.side1.classList.add('on'); if (parts.tog) parts.tog.classList.add('on'); return Promise.resolve(); }
        var target = window.innerWidth > 620 ? parts.side1 : main.querySelector('.zc-h');
        return R.wait(400, id).then(function () { return point(target, id); }).then(function () {
          parts.side1.classList.add('on');
          if (!parts.tog) return R.wait(500, id);
          return point(parts.tog, id).then(function () { parts.tog.classList.add('on'); return R.wait(500, id); });
        });
      },
      function (id, fast) {
        if (fast) { parts.modal.classList.add('on'); return Promise.resolve(); }
        return point(parts.add, id).then(function () { parts.add.classList.add('hot'); return R.wait(250, id); })
          .then(function () { parts.modal.classList.add('on'); cur.style.top = '110%'; return R.wait(500, id); });
      },
      function (id, fast) {
        var p = type(parts.name, 'Zero Mouse Clicks', id, fast);
        return p.then(function () { return fast ? null : R.wait(250, id); }).then(function () { return type(parts.url, url, id, fast); })
          .then(function () { if (parts.chk) { if (fast) { parts.chk.classList.add('on'); return; } return point(parts.chk, id).then(function () { parts.chk.classList.add('on'); }); } })
          .then(function () { return fast ? null : R.wait(300, id); });
      },
      function (id, fast) {
        function fin() { parts.modal.classList.remove('on'); parts.list.style.opacity = '1'; parts.add.classList.remove('hot'); }
        if (fast) { fin(); return Promise.resolve(); }
        return point(parts.ok, id).then(function () { parts.ok.classList.add('hot'); return R.wait(300, id); })
          .then(function () { fin(); cur.style.top = '110%'; return R.wait(2600, id); });
      }
    ];
    var DUR = [2600, 1800, 3400, 3300];
    function play(from) {
      var id = ++R.id; step = from; reset();
      for (var k = 0; k < from; k++) S[k](id, true);
      ol.classList.remove('paused');
      (function next(i) {
        if (i >= S.length) return R.wait(300, id).then(function () { play(0); }, function () {});
        step = i; markSteps(ol, i, DUR[i]);
        return S[i](id, false).then(function () { return next(i + 1); }, function () {});
      })(from);
    }
    function stop() { R.id++; ol.classList.add('paused'); }
    pp.addEventListener('click', function () {
      paused = !paused; pp.textContent = paused ? '▶' : '❚❚'; pp.setAttribute('aria-label', paused ? L.play : L.pause);
      if (paused) stop(); else play(step);
    });
    build();
    if (RM) { reset(); var id0 = ++R.id; S.forEach(function (f) { f(id0, true); }); markSteps(ol, 4, 1); pp.style.display = 'none'; return; }
    visible(root, function (v) { if (v && !on && !paused) { on = true; play(step); } else if (!v && on) { on = false; stop(); } });
  }

  /* ---------------------------------------------------------------- prompt kit */
  function promptkit(root) {
    var L = T[lang(root)], withConn = root.getAttribute('data-mode') === 'connector';
    root.classList.add('zk');
    var f = el('div', 'zk-f'), ins = {};
    L.kFields.forEach(function (d) {
      var lb = el('label', '', esc(d[1])), inp = el('input');
      inp.type = 'text'; inp.autocomplete = 'off'; inp.setAttribute('enterkeyhint', 'next'); inp.placeholder = d[2];
      inp.setAttribute('autocapitalize', 'words'); lb.appendChild(inp); f.appendChild(lb); ins[d[0]] = inp;
    });
    root.appendChild(f);
    var sw = tabs([['no', L.kWithout], ['yes', L.kWith]], withConn ? 'yes' : 'no', function (k) { withConn = k === 'yes'; upd(); }, L.kWith);
    root.appendChild(sw);
    var box = el('div', 'zk-box'); var pre = el('pre'); var code = el('code'); pre.appendChild(code); box.appendChild(pre);
    box.appendChild(copyButton(function () { return text(); }, L)); root.appendChild(box);
    var go = el('div', 'zk-go');
    var aG = el('a', 'pri', esc(L.openG) + ' ↗'), aC = el('a', '', esc(L.openC) + ' ↗');
    [aG, aC].forEach(function (a) { a.target = '_blank'; a.rel = 'noopener'; go.appendChild(a); });
    if (navigator.share) {
      var sh = el('button', '', esc(L.share)); sh.type = 'button';
      sh.addEventListener('click', function () { navigator.share({ text: text() }).catch(function () {}); });
      go.appendChild(sh);
    }
    root.appendChild(go);
    root.appendChild(el('p', 'zk-note', esc(L.kNote)));
    // the prompt is a list of pieces: [text, 0] fixed, [text, 'v'] typed by the user, [text, 'ph'] still to fill
    function parts() {
      var v = {};
      Object.keys(ins).forEach(function (k) { v[k] = ins[k].value.trim().replace(/\s+/g, ' '); });
      var V = function (k) { return v[k] ? [v[k], 'v'] : [L.kPh[k] || '', 'ph']; };
      var pl = L.place(v, V);
      return (withConn ? L.pYes : L.pNo)(pl);
    }
    function text() { return parts().map(function (p) { return p[0]; }).join(''); }
    function upd() {
      code.innerHTML = parts().map(function (p) { return p[1] ? '<span class="' + p[1] + '">' + esc(p[0]) + '</span>' : esc(p[0]); }).join('');
      var t = text();
      aG.href = 'https://chatgpt.com/?' + (withConn ? '' : 'hints=search&') + 'q=' + encodeURIComponent(t);
      aC.href = 'https://claude.ai/new?q=' + encodeURIComponent(t);
    }
    // ?project=…&developer=…&city=…&apt=… fill the form (handy to share a ready link)
    try {
      var q = new URLSearchParams(location.search), map = { proj: 'project', dev: 'developer', city: 'city', apt: 'apt' };
      Object.keys(map).forEach(function (k) { var x = q.get(map[k]); if (x) ins[k].value = x.slice(0, 80); });
    } catch (e) {}
    Object.keys(ins).forEach(function (k) {
      ins[k].addEventListener('input', upd);
      ins[k].addEventListener('keydown', function (e) {
        if (e.key !== 'Enter') return; e.preventDefault();
        var ks = Object.keys(ins), n = ks.indexOf(k);
        if (n < ks.length - 1) ins[ks[n + 1]].focus(); else ins[k].blur();
      });
    });
    upd();
  }

  /* ---------------------------------------------------------------- copy buttons on code blocks */
  function copies() {
    var L = T[lang(document.body)] || T.en;
    [].forEach.call(document.querySelectorAll('.doc pre, pre[data-copy]'), function (pre) {
      if (pre.closest('.zk') || pre.querySelector('.zcp')) return;
      pre.classList.add('zcp-wrap'); pre.style.paddingRight = '56px';
      pre.appendChild(copyButton(function () { var c = pre.querySelector('code'); return (c || pre).textContent.trim(); }, L));
    });
  }

  function init() {
    style();
    [].forEach.call(document.querySelectorAll('[data-howto]'), howto);
    [].forEach.call(document.querySelectorAll('[data-connect]'), connect);
    [].forEach.call(document.querySelectorAll('[data-promptkit]'), promptkit);
    copies();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
