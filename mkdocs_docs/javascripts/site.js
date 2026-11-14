/*
 * palingenesis documentation: behaviour.
 *
 *   1. Reading progress bar (all pages).
 *   2. The title on the landing page: letters that arrange themselves into the name.
 *   3. The docs console on the landing page: one input, four kinds of answer.
 *        Train / Fix   the recipe cards written in index.md
 *        Options       rows of the configuration reference, read from that page
 *        Pages         sections of the docs, read from the search index
 *   4. Math: formulas written as $...$ / $$...$$ (pymdownx.arithmatex) are
 *      typeset by the KaTeX copy in assets/katex, loaded only on pages with math.
 *
 * No dependencies and no network beyond this site. The pages are complete
 * without it: with JavaScript off the recipes are ordinary rows of text.
 */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* -- 1. Reading progress -------------------------------------------------- */

  var bar = document.createElement("div");
  bar.className = "pg-progress";
  bar.setAttribute("aria-hidden", "true");
  document.body.appendChild(bar);

  var ticking = false;
  function updateProgress() {
    ticking = false;
    var max = document.documentElement.scrollHeight - window.innerHeight;
    var read = max > 40 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
    bar.style.setProperty("--pg-read", read.toFixed(4));
  }
  function requestProgress() {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(updateProgress);
    }
  }
  window.addEventListener("scroll", requestProgress, { passive: true });
  window.addEventListener("resize", requestProgress, { passive: true });

  /* -- 2. Title artwork ------------------------------------------------------ */

  /*
   * The name is drawn out of small letters. They start scattered, each one
   * flickering through other characters and tinted; one by one they find their
   * place, settle on the right character and turn to ink, until every large
   * letter is made of its own small ones. Scrolling re-arranges the same
   * letters into even lines of text; the pointer disturbs the ones near it.
   * Everything is a function of time, scroll and pointer: there is no state to
   * drift, so it resizes and reverses cleanly.
   */
  var NOISE = "abcdefghijklmnopqrstuvwxyz0123456789{}[]<>/\\|=+*#%&?!;:.,".split("");

  function smooth(x) {
    x = x < 0 ? 0 : x > 1 ? 1 : x;
    return x * x * (3 - 2 * x);
  }

  function startArt(hero) {
    var canvas = hero.querySelector(".pg-art");
    if (!canvas || !canvas.getContext) return null;
    var ctx = canvas.getContext("2d");
    var word = canvas.getAttribute("data-word") || "";
    var parts = [];
    var width = 0;
    var height = 0;
    var cell = 6;
    var colors = null;
    var frame = 0;
    var running = false;
    var visible = true;
    var alive = true;
    var lastDraw = 0;
    var born = 0;
    var scrollNow = window.scrollY;
    var scrollEase = scrollNow;
    var stir = 0;
    var pointer = { x: -1e4, y: -1e4 };
    var pace = 1;
    try {
      if (sessionStorage.getItem("pg-art")) pace = 0.45; /* seen this session: arrive sooner */
      sessionStorage.setItem("pg-art", "1");
    } catch (e) {}

    function readColors() {
      var style = getComputedStyle(hero);
      colors = {
        ink: style.getPropertyValue("--pg-text").trim() || "#111",
        hues: [
          style.getPropertyValue("--pg-signal").trim() || "#2c4a7c",
          style.getPropertyValue("--pg-signal").trim() || "#2c4a7c",
          style.getPropertyValue("--pg-hue-green").trim() || "#1c6642",
          style.getPropertyValue("--pg-text-3").trim() || "#696964",
        ],
      };
    }

    function build() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.parentElement.clientWidth;
      if (!width || !word) return;
      /* on a phone the small letters shrink to grain, so the name stays crisp */
      cell = Math.max(2.1, Math.min(5.4, width / 240));

      /* fit the word to the width in the heavy sans, then sample it letter by letter */
      var probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
      var family = getComputedStyle(hero).getPropertyValue("--pg-font-sans") || "sans-serif";
      probe.font = "700 200px " + family;
      var tracking = -0.045;
      var advance = [];
      var x = 0;
      for (var i = 0; i < word.length; i++) {
        advance.push(x);
        x += probe.measureText(word[i]).width + tracking * 200;
      }
      var size = Math.min(300, (200 * width) / (x - tracking * 200));
      var scale = size / 200;
      height = Math.round(size * 1.04);
      var baseline = Math.round(size * 0.77);

      canvas.style.height = height + "px";
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      probe.canvas.width = Math.ceil(width);
      probe.canvas.height = height;
      probe.font = "700 " + size + "px " + family;
      probe.textBaseline = "alphabetic";

      var stepX = cell;
      var stepY = cell * 1.28;
      parts = [];
      for (var l = 0; l < word.length; l++) {
        var left = Math.max(0, Math.floor(advance[l] * scale) - 2);
        var span = Math.min(probe.canvas.width - left, Math.ceil(probe.measureText(word[l]).width) + 4);
        if (span <= 0) continue;
        probe.clearRect(0, 0, probe.canvas.width, height);
        probe.fillText(word[l], advance[l] * scale, baseline);
        var data = probe.getImageData(left, 0, span, height).data;
        for (var gy = stepY / 2; gy < height; gy += stepY) {
          for (var gx = left + ((stepX / 2 - left) % stepX + stepX) % stepX; gx < left + span; gx += stepX) {
            var alpha = data[(Math.floor(gy) * span + Math.floor(gx - left)) * 4 + 3];
            if (alpha > 120) parts.push({ tx: gx, ty: gy, glyph: word[l] });
          }
        }
      }

      /* the second arrangement: the same letters set as even lines of text */
      var count = parts.length;
      var pitch = cell * 1.9;
      var cols = Math.max(1, Math.floor(width / pitch));
      var rows = Math.ceil(count / cols);
      var lead = Math.min(cell * 2.4, height / (rows + 1));
      var top = (height - (rows - 1) * lead) / 2;
      var order = parts.map(function (part, index) { return index; }).sort(function (a, b) {
        return parts[a].tx - parts[b].tx || parts[a].ty - parts[b].ty;
      });
      order.forEach(function (index, slot) {
        var part = parts[index];
        var seed = Math.random();
        part.lx = ((slot % cols) + 0.5) * (width / cols);
        part.ly = top + Math.floor(slot / cols) * lead;
        part.line = word[slot % word.length];
        part.cx = Math.random() * width;
        part.cy = Math.random() * height;
        part.phase = seed * Math.PI * 2;
        part.seed = seed;
        part.hue = (Math.random() * 4) | 0;
        /* arrival sweeps left to right, loosely */
        part.delay = 0.25 + 0.9 * (part.tx / width) + Math.random() * 1.1;
        part.key = Math.random();
      });
    }

    function draw(now) {
      frame = 0;
      if (!running) return;
      if (now - lastDraw < 30) {
        frame = requestAnimationFrame(draw);
        return;
      }
      lastDraw = now;
      var still = reduceMotion.matches;
      var t = still ? 99 : (now - born) / 1000;
      if (!colors || ((now / 1000) | 0) % 2 === 0) readColors();

      scrollEase += (scrollNow - scrollEase) * (still ? 1 : 0.18);
      stir *= 0.9;
      var turn = scrollEase / (height * 0.5 + 40); /* 0 = the name, 1 = lines of text */
      var tick = Math.floor(t * 12);

      ctx.clearRect(0, 0, width, height);
      ctx.font = (cell * 1.42).toFixed(2) + 'px "JetBrains Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      var fill = "";
      for (var i = 0; i < parts.length; i++) {
        var q = parts[i];
        var arrive = still ? 1 : smooth((t - q.delay * pace) / (1.5 * pace));
        var arrived = 1 - Math.pow(1 - arrive, 3);
        var to = smooth(turn * 1.7 - q.key * 0.7);
        var mid = Math.sin(Math.PI * to);

        var x = q.tx + (q.lx - q.tx) * to;
        var y = q.ty + (q.ly - q.ty) * to;
        if (arrived < 1) {
          var wx = q.cx + Math.sin(t * 0.7 + q.phase) * 26;
          var wy = q.cy + Math.cos(t * 0.6 + q.phase * 1.7) * 16;
          x = wx + (x - wx) * arrived;
          y = wy + (y - wy) * arrived;
        }
        var loose = mid * 14 + stir * 5;
        if (loose > 0.05) {
          x += Math.sin(q.phase * 7 + t * 2.3) * loose;
          y += Math.cos(q.phase * 5 + t * 1.9) * loose * 0.7;
        }
        var dx = x - pointer.x;
        var dy = y - pointer.y;
        var near = dx * dx + dy * dy;
        var touched = 0;
        if (near < 6400) {
          var d = Math.sqrt(near) || 1;
          touched = 1 - d / 80;
          x += (dx / d) * touched * touched * 26;
          y += (dy / d) * touched * touched * 26;
        }

        /* a letter is settled when it has arrived, is not in transit and is left alone */
        var settled = arrived > 0.9 && mid < 0.35 && touched < 0.25 && stir < 0.5;
        var glyph;
        var color;
        if (settled) {
          glyph = to > 0.5 ? q.line : q.glyph;
          /* now and then one of them thinks again */
          /* a few keep their tint for good; now and then one thinks again */
          color = q.key < 0.09 ? colors.hues[q.hue % 3]
            : !still && ((i * 7919 + Math.floor(t * 3)) % 389 === 0) ? colors.hues[0] : colors.ink;
        } else {
          glyph = NOISE[(((q.seed * 9973) | 0) + tick + i) % NOISE.length];
          color = colors.hues[q.hue];
        }
        if (color !== fill) {
          fill = color;
          ctx.fillStyle = color;
        }
        ctx.globalAlpha = settled ? 1 : 0.35 + 0.65 * Math.max(arrived, 0.2);
        ctx.fillText(glyph, x, y);
      }
      ctx.globalAlpha = 1;

      /* at rest with reduced motion there is nothing more to draw */
      if (still) {
        running = false;
        return;
      }
      frame = requestAnimationFrame(draw);
    }

    function setRunning(on) {
      on = on && alive && visible && !document.hidden && parts.length > 0;
      if (on === running) return;
      running = on;
      if (running) frame = requestAnimationFrame(draw);
      else if (frame) {
        cancelAnimationFrame(frame);
        frame = 0;
      }
    }
    function redraw() {
      running = false;
      setRunning(true);
    }

    function onScroll() {
      var next = window.scrollY;
      stir = Math.min(1, stir + Math.abs(next - scrollNow) / 220);
      scrollNow = next;
      if (reduceMotion.matches) redraw();
    }
    function onPointer(event) {
      if (event.pointerType === "touch") return;
      var box = canvas.getBoundingClientRect();
      pointer.x = event.clientX - box.left;
      pointer.y = event.clientY - box.top;
    }
    function onLeave() {
      pointer.x = pointer.y = -1e4;
    }
    var resizeTimer = 0;
    function onResize() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        if (!alive || canvas.parentElement.clientWidth === width) return;
        build();
        redraw();
      }, 120);
    }
    function onVisibility() {
      setRunning(true);
    }
    var observer = new IntersectionObserver(function (entries) {
      visible = entries[entries.length - 1].isIntersecting;
      setRunning(visible);
    });

    /* the letterforms come from the text face: wait for it, then begin */
    var ready = document.fonts && document.fonts.load
      ? Promise.all([document.fonts.load('700 100px "Geist"'), document.fonts.load('12px "JetBrains Mono"')])
      : Promise.resolve();
    ready.catch(function () {}).then(function () {
      if (!alive) return;
      build();
      if (!parts.length) return;
      hero.classList.add("has-art");
      born = performance.now();
      observer.observe(canvas);
      window.addEventListener("scroll", onScroll, { passive: true });
      window.addEventListener("resize", onResize, { passive: true });
      hero.addEventListener("pointermove", onPointer, { passive: true });
      hero.addEventListener("pointerleave", onLeave, { passive: true });
      document.addEventListener("visibilitychange", onVisibility);
      setRunning(true);
    });

    return function stop() {
      alive = false;
      setRunning(false);
      observer.disconnect();
      clearTimeout(resizeTimer);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }

  /* -- 3. Docs console ------------------------------------------------------- */

  var TAGS = { Train: "run", Fix: "fix", Options: "opt", Pages: "doc" };
  var MAX_OPTIONS = 8;
  var MAX_PAGES = 8;
  var sources = {}; /* url -> promise, kept across instant navigations */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function words(query) {
    return query.toLowerCase().split(/\s+/).filter(Boolean);
  }

  /* Every word must occur; a hit in the title outranks a hit in the body. */
  function score(terms, title, body) {
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var term = terms[i];
      var at = title.indexOf(term);
      if (at === 0) total += 12;
      else if (at > 0) total += 8;
      else if (body.indexOf(term) >= 0) total += 1;
      else return 0;
    }
    return total;
  }

  function loadOnce(url, parse) {
    url = new URL(url, document.baseURI).href;
    if (!sources[url]) {
      sources[url] = fetch(url, { credentials: "same-origin" })
        .then(function (response) {
          if (!response.ok) throw new Error(response.status + " " + url);
          return response.text();
        })
        .then(function (text) {
          return parse(text, url);
        })
        .catch(function () {
          delete sources[url];
          return [];
        });
    }
    return sources[url];
  }

  function headingText(heading) {
    var copy = heading.cloneNode(true);
    var link = copy.querySelector(".headerlink");
    if (link) link.remove();
    return copy.textContent.trim();
  }

  /* Rows of every "Parameter | Type | Default | Description" table on the page. */
  function parseOptions(html, url) {
    var page = new DOMParser().parseFromString(html, "text/html");
    var out = [];
    page.querySelectorAll("table").forEach(function (table) {
      var head = [].map.call(table.querySelectorAll("thead th"), function (th) {
        return th.textContent.trim().toLowerCase();
      });
      if (head[0] !== "parameter") return;
      var typeAt = head.indexOf("type");
      var defaultAt = head.indexOf("default");
      var descAt = head.indexOf("description");

      var heading = null;
      var parent = null;
      for (var node = table.previousElementSibling; node; node = node.previousElementSibling) {
        if (!heading && /^H[23]$/.test(node.tagName)) {
          heading = node;
          if (node.tagName === "H2") break;
        } else if (heading && node.tagName === "H2") {
          parent = node;
          break;
        }
      }
      if (!heading) return;
      var section = headingText(heading);

      table.querySelectorAll("tbody tr").forEach(function (row) {
        var cells = row.children;
        if (!cells.length) return;
        var codes = cells[0].querySelectorAll("code");
        var name = (codes.length === 1 ? codes[0] : cells[0]).textContent.trim();
        if (!name) return;
        var desc = descAt >= 0 && cells[descAt] ? cells[descAt].cloneNode(true) : null;
        if (desc) {
          desc.querySelectorAll("a[href]").forEach(function (a) {
            a.setAttribute("href", new URL(a.getAttribute("href"), url).href);
          });
        }
        var descText = desc ? desc.textContent : "";
        out.push({
          kind: "option",
          group: "Options",
          title: section + "." + name,
          hint: typeAt >= 0 && cells[typeAt] ? cells[typeAt].textContent.trim() : "",
          name: name,
          section: section,
          parent: parent ? headingText(parent) : "",
          simple: codes.length === 1 && /^[\w.]+$/.test(name) && /^[a-z_]+$/.test(section),
          def: defaultAt >= 0 && cells[defaultAt] ? cells[defaultAt].textContent.trim() : "",
          desc: desc,
          href: new URL("#" + heading.id, url).href,
          titleKey: (name + " " + section + "." + name).toLowerCase(),
          bodyKey: descText.toLowerCase(),
        });
      });
    });
    return out;
  }

  function parsePages(text, url) {
    var data = JSON.parse(text);
    var base = new URL("..", url); /* the index lives in <site>/search/ */
    var pageTitle = {};
    data.docs.forEach(function (doc) {
      if (doc.location.indexOf("#") < 0) pageTitle[doc.location] = doc.title;
    });
    return data.docs
      .filter(function (doc) {
        /* skip the landing page itself: its sections are the recipes above */
        return doc.title && doc.location && doc.location.charAt(0) !== "#";
      })
      .map(function (doc) {
        var path = doc.location.split("#")[0];
        var plain = doc.text.replace(/<[^>]*>/g, " ");
        return {
          kind: "page",
          group: "Pages",
          title: doc.title,
          hint: pageTitle[path] && pageTitle[path] !== doc.title ? pageTitle[path] : "",
          html: doc.text,
          href: new URL(doc.location, base).href,
          titleKey: doc.title.toLowerCase(),
          bodyKey: plain.toLowerCase(),
        };
      });
  }

  function startConsole(root) {
    var input = root.querySelector(".pg-console__input");
    var list = root.querySelector(".pg-console__list");
    var view = root.querySelector(".pg-console__view");
    var status = root.querySelector(".pg-console__status");
    var cards = document.querySelectorAll(".pg-home .pg-recipe");
    if (!input || !list || !view || !cards.length) return null;

    var stash = el("div");
    stash.hidden = true;
    root.appendChild(stash);

    var recipes = [].map.call(cards, function (card) {
      var heading = card.querySelector("h3");
      var command = card.querySelector("pre code");
      var title = heading ? headingText(heading) : "";
      var group = card.getAttribute("data-group") || "Train";
      stash.appendChild(card);
      return {
        kind: "recipe",
        group: group,
        title: title,
        hint: group === "Train" && command ? command.textContent.trim().split(/\s+/).slice(0, 2).join(" ") : "",
        node: card,
        titleKey: (title + " " + (card.getAttribute("data-keys") || "")).toLowerCase(),
        bodyKey: card.textContent.toLowerCase(),
      };
    });
    document.querySelectorAll(".pg-home .pg-source").forEach(function (section) {
      section.hidden = true;
    });

    var options = [];
    var pages = [];
    var items = [];
    var selected = -1;
    var shown = null;
    var alive = true;

    function load() {
      loadOnce(root.getAttribute("data-config"), parseOptions).then(function (rows) {
        options = rows;
        if (alive && input.value.trim()) render(true);
      });
      loadOnce(root.getAttribute("data-index"), parsePages).then(function (rows) {
        pages = rows;
        if (alive && input.value.trim()) render(true);
      });
    }

    function rank(pool, terms, limit) {
      var hits = [];
      for (var i = 0; i < pool.length; i++) {
        var s = score(terms, pool[i].titleKey, pool[i].bodyKey);
        if (s) hits.push({ item: pool[i], score: s, order: i });
      }
      hits.sort(function (a, b) {
        return b.score - a.score || a.order - b.order;
      });
      if (limit) hits.length = Math.min(hits.length, limit);
      return hits.map(function (hit) {
        return hit.item;
      });
    }

    function render(keep) {
      var terms = words(input.value);
      var previous = keep && selected >= 0 ? items[selected] : null;
      var groups = [];

      if (!terms.length) {
        ["Train", "Fix"].forEach(function (name) {
          groups.push([name, recipes.filter(function (r) { return r.group === name; })]);
        });
      } else {
        ["Train", "Fix"].forEach(function (name) {
          groups.push([name, rank(recipes.filter(function (r) { return r.group === name; }), terms)]);
        });
        groups.push(["Options", rank(options, terms, MAX_OPTIONS)]);
        groups.push(["Pages", rank(pages, terms, MAX_PAGES)]);
        /* an exact option name is what the reader typed for: show options first */
        var top = groups[2][1][0];
        if (top && terms.length === 1 && top.name.toLowerCase().indexOf(terms[0]) === 0) {
          groups.unshift(groups.splice(2, 1)[0]);
        }
      }

      items = [];
      list.textContent = "";
      groups.forEach(function (group) {
        if (!group[1].length) return;
        var head = el("div", "pg-group");
        head.setAttribute("role", "presentation");
        head.appendChild(el("span", "", group[0]));
        head.appendChild(el("span", "pg-group__n", String(group[1].length)));
        list.appendChild(head);
        group[1].forEach(function (item) {
          var index = items.length;
          var option = el("div", "pg-opt pg-opt--" + item.kind + " pg-opt--" + item.group.toLowerCase());
          option.id = "pg-opt-" + index;
          option.setAttribute("role", "option");
          option.setAttribute("aria-selected", "false");
          option.appendChild(el("span", "pg-opt__tag", TAGS[item.group]));
          option.appendChild(el("span", "pg-opt__label", item.title));
          if (item.hint) option.appendChild(el("span", "pg-opt__hint", item.hint));
          option.addEventListener("click", function () {
            select(index, false);
            if (window.matchMedia("(max-width: 44.9844em)").matches) {
              view.scrollIntoView({ block: "nearest", behavior: reduceMotion.matches ? "auto" : "smooth" });
            }
          });
          option.addEventListener("dblclick", open);
          list.appendChild(option);
          items.push(item);
        });
      });

      root.classList.toggle("is-empty", !items.length);
      if (!items.length) {
        selected = -1;
        shown = null;
        input.removeAttribute("aria-activedescendant");
        view.textContent = "";
        var none = el("div", "pg-view__none");
        none.appendChild(el("p", "", "Nothing matches “" + input.value.trim() + "”."));
        none.appendChild(el("p", "", "Try a method (distill, DPO), a symptom (NaN, slow) or an option name (packing)."));
        view.appendChild(none);
        status.textContent = "No answers";
        return;
      }
      status.textContent = items.length + (items.length === 1 ? " answer" : " answers");
      var at = previous ? items.indexOf(previous) : -1;
      select(at >= 0 ? at : 0, false);
      if (at < 0) list.scrollTop = 0;
    }

    function select(index, scroll) {
      if (!items.length) return;
      selected = (index + items.length) % items.length;
      var nodes = list.querySelectorAll(".pg-opt");
      for (var i = 0; i < nodes.length; i++) {
        nodes[i].setAttribute("aria-selected", i === selected ? "true" : "false");
      }
      input.setAttribute("aria-activedescendant", "pg-opt-" + selected);
      if (scroll) nodes[selected].scrollIntoView({ block: "nearest" });
      show(items[selected]);
    }

    function crumb(parts) {
      var p = el("p", "pg-view__crumb");
      parts.filter(Boolean).forEach(function (part) {
        p.appendChild(el("span", "", part));
      });
      return p;
    }

    function more(href, label) {
      var p = el("p", "pg-more");
      var a = el("a", "", label);
      a.href = href;
      p.appendChild(a);
      return p;
    }

    function code(language, text) {
      var box = el("div", "highlight language-" + language);
      var pre = el("pre");
      pre.appendChild(el("code", "", text));
      box.appendChild(pre);
      return box;
    }

    function optionView(item) {
      var box = el("article", "pg-answer");
      box.appendChild(crumb(["Configuration", item.parent, item.section]));
      var h = el("h3");
      h.appendChild(el("code", "", item.title));
      box.appendChild(h);
      var meta = el("p", "pg-answer__meta");
      [["type", item.hint], ["default", item.def]].forEach(function (pair) {
        if (!pair[1]) return;
        var chip = el("span", "", pair[0] + " ");
        chip.appendChild(el("code", "", pair[1]));
        meta.appendChild(chip);
      });
      if (meta.childNodes.length) box.appendChild(meta);
      if (item.desc) {
        var desc = el("p", "pg-answer__desc");
        [].forEach.call(item.desc.cloneNode(true).childNodes, function (node) {
          desc.appendChild(node.cloneNode(true));
        });
        box.appendChild(desc);
      }
      if (item.simple) {
        var value = /^[^\s#]+$/.test(item.def) ? item.def : "...";
        box.appendChild(code("yaml", item.section + ":\n  " + item.name + ": " + value));
        var cli = el("p", "pg-answer__cli", "From the command line: ");
        cli.appendChild(el("code", "", "--" + item.section + "." + item.name + " VALUE"));
        box.appendChild(cli);
      }
      box.appendChild(more(item.href, "Open in the configuration reference"));
      return box;
    }

    function pageView(item) {
      var box = el("article", "pg-answer");
      box.appendChild(crumb(["Docs", item.hint]));
      box.appendChild(el("h3", "", item.title));
      var text = new DOMParser().parseFromString(item.html, "text/html").body.textContent.replace(/\s+/g, " ").trim();
      var terms = words(input.value);
      var lower = text.toLowerCase();
      var first = -1;
      terms.forEach(function (term) {
        var at = lower.indexOf(term);
        if (at >= 0 && (first < 0 || at < first)) first = at;
      });
      var from = first > 90 ? first - 70 : 0;
      var excerpt = (from ? "…" : "") + text.slice(from, from + 330) + (from + 330 < text.length ? "…" : "");
      var p = el("p", "pg-answer__desc");
      var pattern = terms.length
        ? new RegExp("(" + terms.map(function (t) { return t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }).join("|") + ")", "gi")
        : null;
      (pattern ? excerpt.split(pattern) : [excerpt]).forEach(function (piece, i) {
        if (!piece) return;
        p.appendChild(i % 2 ? el("mark", "", piece) : document.createTextNode(piece));
      });
      if (text) box.appendChild(p);
      box.appendChild(more(item.href, "Open this section"));
      return box;
    }

    function show(item) {
      if (shown === item) return;
      if (shown && shown.kind === "recipe") stash.appendChild(shown.node);
      shown = item;
      view.textContent = "";
      view.scrollTop = 0;
      view.appendChild(item.kind === "recipe" ? item.node : item.kind === "option" ? optionView(item) : pageView(item));
      /* restart the entrance animation of the answer */
      view.classList.remove("is-fresh");
      void view.offsetWidth;
      view.classList.add("is-fresh");
    }

    function open() {
      var link = view.querySelector(".pg-more a") || view.querySelector("a[href]");
      if (link) link.click();
    }

    function setQuery(value) {
      input.value = value;
      render(false);
    }

    input.addEventListener("input", function () {
      load();
      render(false);
    });
    input.addEventListener("focus", load);
    input.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        select(selected + (event.key === "ArrowDown" ? 1 : -1), true);
      } else if (event.key === "Enter") {
        event.preventDefault();
        open();
      } else if (event.key === "Escape" && input.value) {
        event.preventDefault();
        setQuery("");
      }
    });
    root.querySelectorAll(".pg-console__try button").forEach(function (button) {
      button.addEventListener("click", function () {
        load();
        setQuery(button.getAttribute("data-q"));
        if (window.matchMedia("(hover: hover)").matches) input.focus();
      });
    });
    render(false);
    root.classList.add("is-live");

    var idle = window.requestIdleCallback || function (run) { return setTimeout(run, 600); };
    idle(function () {
      if (alive) load();
    });

    return function stop() {
      alive = false;
    };
  }

  /* -- 4. Math ---------------------------------------------------------------- */

  var here = document.currentScript && document.currentScript.src;
  var katexUrl = here ? here.replace(/javascripts\/site\.js(\?.*)?$/, "assets/katex/katex.min.js") : null;
  var katexReady = null;

  function loadKatex() {
    if (window.katex) return Promise.resolve(window.katex);
    if (!katexReady) {
      katexReady = new Promise(function (resolve, reject) {
        var script = document.createElement("script");
        script.src = katexUrl;
        script.async = true;
        script.onload = function () { resolve(window.katex); };
        script.onerror = function () { katexReady = null; reject(new Error("KaTeX did not load")); };
        document.head.appendChild(script);
      });
    }
    return katexReady;
  }

  function typesetMath() {
    var nodes = document.querySelectorAll(".arithmatex:not(.is-rendered)");
    if (!nodes.length || !katexUrl) return;
    loadKatex().then(function (katex) {
      nodes.forEach(function (node) {
        var source = node.textContent.trim();
        var display = node.tagName === "DIV";
        var tex = source.replace(/^\\[(\[]/, "").replace(/\\[)\]]$/, "");
        katex.render(tex, node, { displayMode: display, throwOnError: false, strict: "ignore" });
        node.setAttribute("data-tex", tex);
        node.classList.add("is-rendered");
      });
    }).catch(function () {
      /* Without KaTeX the source stays readable. */
      nodes.forEach(function (node) { node.classList.add("is-rendered"); });
    });
  }

  /* -- Wiring: once now, and again after each instant navigation ------------- */

  var stopConsole = null;
  var stopArt = null;
  function onPage() {
    if (stopConsole) stopConsole();
    if (stopArt) stopArt();
    stopConsole = stopArt = null;
    var root = document.querySelector(".pg-console");
    if (root && !root.classList.contains("is-live")) stopConsole = startConsole(root);
    var hero = document.querySelector(".pg-hero");
    if (hero && !hero.classList.contains("has-art")) stopArt = startArt(hero);
    typesetMath();
    requestProgress();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(onPage);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onPage);
  } else {
    onPage();
  }
})();
