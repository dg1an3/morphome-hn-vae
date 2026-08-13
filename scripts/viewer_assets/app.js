/* morphome synthetic corpus explorer.
 *
 * Left: cascaded-UMAP scatters (2D clickable, 3D drag-to-rotate), real and
 * synthetic co-embedded. Right: vtk.js slices of the selected case, loaded on
 * demand from data/vol/<case>_{ct,seg}.bin, in either of two layouts --
 *
 *   mode3d  the three planes as intersecting slices in one rotatable 3D scene
 *   panes   the three planes as separate parallel-projection views
 *
 * -- sharing one set of slice indices, window/level and organ overlay.
 *
 * Volumes are raw uint8, written from numpy (z,y,x) C-order, which is already
 * VTK's x-fastest layout: K=transverse, J=coronal, I=sagittal. The grid is not
 * assumed cubic -- the 2.5 mm dose frame is 192x128x96 -- so dims come from
 * embedding.json and every per-plane extent is taken from its own axis. This
 * bundle's SlicingMode is {I:0, J:1, K:2}, so a view's mode indexes DIMS
 * directly: mode 0 slices x, 1 slices y, 2 slices z.
 */
(function () {
  "use strict";

  var DIMS = [128, 128, 128], SPACING = 1.6;
  var statusEl = document.getElementById("status");
  var metaEl = document.getElementById("meta");
  var data = null, sel = -1, views = {}, volCache = {};
  var v3 = null, SM = null, curImg = null, homed3 = false;
  var KEYS = ["K", "J", "I"], PLANE_LABEL = { K: "T", J: "C", I: "S" };

  function status(msg, warn) {
    statusEl.textContent = msg;
    statusEl.className = warn ? "warn" : "";
  }

  // ---------------------------------------------------------------- colours
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    var stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    var f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
    var a = stops[i], b = stops[i + 1];
    return "rgb(" + Math.round(a[0] + (b[0] - a[0]) * u) + "," +
      Math.round(a[1] + (b[1] - a[1]) * u) + "," +
      Math.round(a[2] + (b[2] - a[2]) * u) + ")";
  }

  function colourOf(rec) {
    var mode = document.getElementById("colorBy").value;
    if (mode === "corpus") return rec.real ? "#f0883e" : "#58a6ff";
    var vals = data.cases.map(function (c) { return c[mode]; })
      .filter(function (v) { return typeof v === "number" && isFinite(v); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var v = rec[mode];
    return (typeof v !== "number" || !isFinite(v)) ? "#555"
      : viridis(hi > lo ? (v - lo) / (hi - lo) : 0.5);
  }

  // ------------------------------------------------------------- 2D scatter
  var c2 = document.getElementById("sc2"), pts2 = [];

  function fit(canvas) {
    var r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: r.width, h: r.height };
  }

  function draw2() {
    var f = fit(c2), ctx = f.ctx, pad = 14;
    ctx.clearRect(0, 0, f.w, f.h);
    var xs = data.cases.map(function (c) { return c.umap2[0]; });
    var ys = data.cases.map(function (c) { return c.umap2[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    pts2 = data.cases.map(function (rec) {
      return {
        x: pad + (rec.umap2[0] - x0) / (x1 - x0 || 1) * (f.w - 2 * pad),
        y: f.h - pad - (rec.umap2[1] - y0) / (y1 - y0 || 1) * (f.h - 2 * pad)
      };
    });
    data.cases.forEach(function (rec, i) {
      var p = pts2[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, rec.real ? 4 : 3, 0, 6.2832);
      ctx.fillStyle = colourOf(rec);
      ctx.globalAlpha = i === sel ? 1 : 0.82;
      ctx.fill();
      if (rec.real) { ctx.lineWidth = 1; ctx.strokeStyle = "#0b0e13"; ctx.stroke(); }
      if (i === sel) {
        ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = "#fff";
        ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, 6.2832); ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------- 3D scatter
  var c3 = document.getElementById("sc3"), yaw = 0.6, pitch = -0.35, pts3 = [];

  function draw3() {
    var f = fit(c3), ctx = f.ctx, pad = 18;
    ctx.clearRect(0, 0, f.w, f.h);
    var mins = [0, 1, 2].map(function (k) {
      return Math.min.apply(null, data.cases.map(function (c) { return c.umap3[k]; }));
    });
    var maxs = [0, 1, 2].map(function (k) {
      return Math.max.apply(null, data.cases.map(function (c) { return c.umap3[k]; }));
    });
    var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    var proj = data.cases.map(function (rec) {
      var v = [0, 1, 2].map(function (k) {
        return (rec.umap3[k] - mins[k]) / ((maxs[k] - mins[k]) || 1) * 2 - 1;
      });
      var x = v[0] * cy - v[2] * sy, z = v[0] * sy + v[2] * cy;
      var y = v[1] * cp - z * sp, depth = v[1] * sp + z * cp;
      return { x: x, y: y, d: depth };
    });
    var s = Math.min(f.w, f.h) / 2 - pad;
    pts3 = proj.map(function (p) { return { x: f.w / 2 + p.x * s, y: f.h / 2 - p.y * s }; });
    var order = proj.map(function (p, i) { return i; })
      .sort(function (a, b) { return proj[a].d - proj[b].d; });
    order.forEach(function (i) {
      var rec = data.cases[i], p = pts3[i];
      var t = (proj[i].d + 1.8) / 3.6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, (rec.real ? 4 : 3) * (0.65 + 0.5 * t), 0, 6.2832);
      ctx.fillStyle = colourOf(rec);
      ctx.globalAlpha = 0.35 + 0.6 * t;
      ctx.fill();
      if (i === sel) {
        ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = "#fff";
        ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, 6.2832); ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }

  function nearest(pts, ev, canvas) {
    var r = canvas.getBoundingClientRect();
    var mx = ev.clientX - r.left, my = ev.clientY - r.top, best = -1, bd = 1e9;
    pts.forEach(function (p, i) {
      var d = (p.x - mx) * (p.x - mx) + (p.y - my) * (p.y - my);
      if (d < bd) { bd = d; best = i; }
    });
    return bd < 400 ? best : -1;
  }

  // ------------------------------------------------------------- vtk views
  // Edge letters make the orientation falsifiable at a glance instead of
  // resting on the camera algebra being right.
  function addOrientLabels(containerId, mode) {
    var host = document.getElementById(containerId).parentNode;
    var letters = mode === 2 ? ["A", "P", "R", "L"]
      : (mode === 1 ? ["S", "I", "R", "L"] : ["S", "I", "A", "P"]);
    ["top", "bottom", "left", "right"].forEach(function (pos, i) {
      var el = document.createElement("div");
      el.className = "orient " + pos;
      el.textContent = letters[i];
      host.appendChild(el);
    });
  }

  // A CT slice plus its coregistered organ overlay, added to `ren`. Both
  // layouts are built from this, so a plane behaves identically whether it is
  // alone in a parallel-projection pane or intersecting two others in 3D.
  function slicePair(ren, mode) {
    var C = vtk.Rendering.Core;
    var mapper = C.vtkImageMapper.newInstance();
    mapper.setSlicingMode(mode);
    var actor = C.vtkImageSlice.newInstance();
    actor.setMapper(mapper);
    ren.addActor(actor);

    var segMapper = C.vtkImageMapper.newInstance();
    segMapper.setSlicingMode(mode);
    var segActor = C.vtkImageSlice.newInstance();
    segActor.setMapper(segMapper);

    // 0 stays transparent so background never paints over the CT.
    var ctf = C.vtkColorTransferFunction.newInstance();
    var pal = [[0, 0, 0], [0.90, 0.10, 0.10], [1.00, 0.95, 0.20], [0.20, 0.60, 1.00],
      [0.20, 1.00, 0.40], [0.00, 0.70, 0.25], [1.00, 0.45, 0.00],
      [0.85, 0.30, 0.00], [0.80, 0.30, 1.00], [0.55, 0.10, 0.85]];
    pal.forEach(function (c, i) { ctf.addRGBPoint(i, c[0], c[1], c[2]); });
    var pwf = vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();
    pwf.addPoint(0, 0.0); pwf.addPoint(0.5, 0.0); pwf.addPoint(1, 0.45); pwf.addPoint(9, 0.45);
    var sp = segActor.getProperty();
    sp.setRGBTransferFunction(0, ctf);
    if (sp.setPiecewiseFunction) sp.setPiecewiseFunction(0, pwf);
    else if (sp.setScalarOpacity) sp.setScalarOpacity(0, pwf);
    if (sp.setInterpolationType) sp.setInterpolationType(0);   // nearest
    // The property normalises scalars through colorWindow/colorLevel *before*
    // the transfer-function lookup. Left at the 255/127.5 default, labels 1..9
    // all collapse to nearly the same colour; the window has to match the label
    // range for the palette to survive.
    sp.setColorWindow(9);
    sp.setColorLevel(4.5);

    ren.addActor(segActor);
    return { mapper: mapper, actor: actor, segMapper: segMapper, segActor: segActor,
             mode: mode };
  }

  function makeView(id, mode, idxEl) {
    var M = vtk.Rendering.Misc;
    addOrientLabels(id, mode);
    var grw = M.vtkGenericRenderWindow.newInstance({ background: [0.04, 0.05, 0.07] });
    grw.setContainer(document.getElementById(id));
    grw.resize();
    var ren = grw.getRenderer(), rw = grw.getRenderWindow();
    var p = slicePair(ren, mode);
    ren.getActiveCamera().setParallelProjection(true);
    return { grw: grw, ren: ren, rw: rw, mapper: p.mapper, actor: p.actor,
             segMapper: p.segMapper, segActor: p.segActor, mode: mode, idxEl: idxEl };
  }

  // The three planes need no placement of their own: vtkImageMapper already
  // emits each slice as a quad at its true world position, so dropping all
  // three into one renderer puts them where they physically intersect. What
  // differs from a 2D pane is the camera -- perspective, trackball-driven, and
  // homed once rather than reset per case, so a rotation survives clicking
  // through the corpus.
  function make3D(id) {
    var M = vtk.Rendering.Misc;
    var grw = M.vtkGenericRenderWindow.newInstance({ background: [0.04, 0.05, 0.07] });
    grw.setContainer(document.getElementById(id));
    grw.resize();
    var ren = grw.getRenderer(), rw = grw.getRenderWindow();
    var iren = grw.getInteractor();
    iren.setInteractorStyle(
      vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance());

    var planes = {};
    [["K", 2], ["J", 1], ["I", 0]].forEach(function (kv) {
      planes[kv[0]] = slicePair(ren, kv[1]);
    });
    ren.getActiveCamera().setParallelProjection(false);
    addCube(ren, iren);
    return { grw: grw, ren: ren, rw: rw, planes: planes };
  }

  // An orientation cube is to the 3D view what the edge letters are to the 2D
  // panes: it makes handedness falsifiable instead of resting on the camera
  // algebra being right. Faces follow LPS -- +x left, +y posterior, +z superior.
  function addCube(ren, iren) {
    var C = vtk.Rendering.Core, W = vtk.Interaction.Widgets;
    if (!C.vtkAnnotatedCubeActor || !W || !W.vtkOrientationMarkerWidget) return;
    try {
      var cube = C.vtkAnnotatedCubeActor.newInstance();
      cube.setDefaultStyle({
        fontFamily: "Arial", fontColor: "#e6edf3", fontSizeScale: function (r) { return r / 2; },
        faceColor: "#1d232c", edgeThickness: 0.08, edgeColor: "#2a3038", resolution: 256
      });
      // The face setters replace the whole property object rather than merging,
      // so the per-face rotation has to be restated with the text or every
      // letter but A renders on its side.
      [["setXPlusFaceProperty", "L", 90], ["setXMinusFaceProperty", "R", -90],
       ["setYPlusFaceProperty", "P", 180], ["setYMinusFaceProperty", "A", 0],
       ["setZPlusFaceProperty", "S", 0], ["setZMinusFaceProperty", "I", 180]]
        .forEach(function (f) { cube[f[0]]({ text: f[1], faceRotation: f[2] }); });
      var w = W.vtkOrientationMarkerWidget.newInstance({ actor: cube, interactor: iren });
      w.setEnabled(true);
      w.setViewportCorner(W.vtkOrientationMarkerWidget.Corners.BOTTOM_RIGHT);
      w.setViewportSize(0.14);
    } catch (e) {
      // Cosmetic only -- never let it take the scene down with it.
      console.warn("orientation cube unavailable:", e);
    }
  }

  function eachPair(fn) {
    KEYS.forEach(function (k) { if (views[k]) fn(views[k], k); });
    if (v3) KEYS.forEach(function (k) { fn(v3.planes[k], k); });
  }

  function renderAll() {
    KEYS.forEach(function (k) { if (views[k]) views[k].rw.render(); });
    if (v3) v3.rw.render();
  }

  function imageDataFrom(bytes) {
    var img = vtk.Common.DataModel.vtkImageData.newInstance();
    img.setDimensions(DIMS[0], DIMS[1], DIMS[2]);
    img.setSpacing(SPACING, SPACING, SPACING);
    img.setOrigin(0, 0, 0);
    var da = vtk.Common.Core.vtkDataArray.newInstance({
      name: "scalars", numberOfComponents: 1, values: bytes });
    img.getPointData().setScalars(da);
    return img;
  }

  function orientCamera(v) {
    var c = v.ren.getActiveCamera();
    // Centre each axis on its own extent. A single `mid` only lands in the
    // middle of the volume when the grid is cubic; on 192x128x96 it would put
    // the focal point outside the short axis entirely.
    var cx = (DIMS[0] - 1) * SPACING / 2;
    var cy = (DIMS[1] - 1) * SPACING / 2;
    var cz = (DIMS[2] - 1) * SPACING / 2;
    var d = Math.max(DIMS[0], DIMS[1], DIMS[2]) * SPACING * 2;
    // Volumes are LPS: +x left, +y posterior, +z superior. Radiological
    // convention, which is what these must match:
    //   transverse  viewed from BELOW  -> anterior up, patient right screen-left
    //   coronal     viewed from FRONT  -> superior up, patient right screen-left
    //   sagittal    viewed from LEFT   -> superior up, anterior screen-left
    // Camera side and viewUp are not independent: viewUp alone rotates the image
    // 180 degrees, flipping left-right along with up-down, so each plane needs
    // both set together.
    if (v.mode === 2) { c.setPosition(cx, cy, cz - d); c.setViewUp(0, -1, 0); }
    else if (v.mode === 1) { c.setPosition(cx, cy - d, cz); c.setViewUp(0, 0, 1); }
    else { c.setPosition(cx + d, cy, cz); c.setViewUp(0, 0, 1); }
    c.setFocalPoint(cx, cy, cz);
    v.ren.resetCamera();
    // resetCamera fits the whole 3D bounding box, which leaves a slice framed
    // inside a lot of empty pane. Fit the in-plane extent instead -- the two
    // axes this plane does *not* slice, so an anisotropic grid still fills it.
    var ex = (DIMS[0] - 1) * SPACING, ey = (DIMS[1] - 1) * SPACING,
        ez = (DIMS[2] - 1) * SPACING;
    var inPlane = v.mode === 2 ? Math.max(ex, ey)
      : (v.mode === 1 ? Math.max(ex, ez) : Math.max(ey, ez));
    c.setParallelScale(inPlane / 2 * 1.02);
  }

  // The 3D camera is homed on the first volume and then left alone, so a
  // rotation the user set survives selecting another case; `force` is the
  // reset-view button.
  function orient3D(force) {
    if (!v3 || (homed3 && !force)) return;
    var c = v3.ren.getActiveCamera();
    var cx = (DIMS[0] - 1) * SPACING / 2;
    var cy = (DIMS[1] - 1) * SPACING / 2;
    var cz = (DIMS[2] - 1) * SPACING / 2;
    var d = Math.max(DIMS[0], DIMS[1], DIMS[2]) * SPACING * 1.6;
    // Right-anterior-superior oblique. With viewUp +z the screen-right vector
    // works out to +x/-y, i.e. patient left to the right of screen and anterior
    // toward the viewer -- the same handedness the transverse and coronal panes
    // present, so switching layouts does not mirror the anatomy.
    c.setFocalPoint(cx, cy, cz);
    c.setPosition(cx - 0.8 * d, cy - 0.8 * d, cz + 0.5 * d);
    c.setViewUp(0, 0, 1);
    // resetCamera fits the planes' full bounding sphere, which on this grid
    // leaves the anatomy sitting in a lot of empty pane; the planes never leave
    // the frustum at this factor.
    v3.ren.resetCamera();
    c.zoom(1.35);
    v3.ren.resetCameraClippingRange();
    homed3 = true;
    v3.rw.render();
  }

  function applyWL() {
    var w = +document.getElementById("win").value, l = +document.getElementById("lev").value;
    eachPair(function (p) {
      p.actor.getProperty().setColorWindow(w);
      p.actor.getProperty().setColorLevel(l);
    });
    renderAll();
  }

  // Organ overlay is global; per-plane visibility is a 3D-only control, since
  // hiding a plane in the pane layout would just blank one of the three views.
  function applyVis() {
    var seg = document.getElementById("seg").checked;
    KEYS.forEach(function (k) { if (views[k]) views[k].segActor.setVisibility(seg); });
    if (v3) KEYS.forEach(function (k) {
      var on = document.getElementById("p" + k).checked;
      v3.planes[k].actor.setVisibility(on);
      v3.planes[k].segActor.setVisibility(on && seg);
    });
    renderAll();
  }

  function setSlice(key, n) {
    var v = views[key];
    if (v) {
      v.mapper.setSlice(n);
      v.segMapper.setSlice(n);
      if (v.idxEl) v.idxEl.textContent = n + " / " + (DIMS[v.mode] - 1);
      v.rw.render();
    }
    if (v3) {
      v3.planes[key].mapper.setSlice(n);
      v3.planes[key].segMapper.setSlice(n);
      document.getElementById("i3").textContent = KEYS.map(function (k) {
        return PLANE_LABEL[k] + " " + document.getElementById("sl" + k).value;
      }).join("   ");
      v3.rw.render();
    }
  }

  function fetchVol(name, kind) {
    var key = name + "_" + kind;
    if (volCache[key]) return Promise.resolve(volCache[key]);
    return fetch("data/vol/" + key + ".bin").then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + key);
      return r.arrayBuffer();
    }).then(function (buf) {
      var arr = new Uint8Array(buf);
      volCache[key] = arr;
      return arr;
    });
  }

  function select(i) {
    if (i < 0 || i === sel) return;
    sel = i;
    var rec = data.cases[i];
    showMeta(rec);
    draw2(); draw3();
    status("loading " + rec.name + " …");
    Promise.all([fetchVol(rec.name, "ct"), fetchVol(rec.name, "seg")])
      .then(function (r) {
        curImg = { ct: imageDataFrom(r[0]), sg: imageDataFrom(r[1]) };
        eachPair(function (p) {
          p.mapper.setInputData(curImg.ct);
          p.segMapper.setInputData(curImg.sg);
        });
        KEYS.forEach(function (k) { if (views[k]) orientCamera(views[k]); });
        orient3D(false);
        applyWL();
        applyVis();
        KEYS.forEach(function (k) {
          setSlice(k, +document.getElementById("sl" + k).value);
        });
        status(rec.name + (rec.real ? "  (real case)" : "  (synthetic)"));
      })
      .catch(function (e) { status("failed to load volume: " + e.message, true); });
  }

  function showMeta(rec) {
    var rows = [
      ["case", rec.name], ["corpus", rec.real ? "real (PDDCA)" : "synthetic"],
      ["organs present", rec.n_present + " / 9"],
      ["bone sharpness", (rec.sharpness || 0).toFixed(1)],
      ["body fraction", (rec.body_frac || 0).toFixed(3)]
    ];
    var html = "<h2 style='font-size:11px;text-transform:uppercase;letter-spacing:.08em;" +
      "color:var(--dim);margin:0 0 8px'>Selected case</h2><table>";
    rows.forEach(function (r) { html += "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td></tr>"; });
    html += "</table><h2 style='font-size:11px;text-transform:uppercase;letter-spacing:.08em;" +
      "color:var(--dim);margin:14px 0 8px'>Organ voxels</h2><table>";
    data.structures.forEach(function (s) {
      html += "<tr><td>" + s + "</td><td>" +
        (rec.organ_voxels[s] || 0).toLocaleString() + "</td></tr>";
    });
    metaEl.innerHTML = html + "</table>";
  }

  // ---------------------------------------------------------------- layout
  // The panes are built the first time they are shown rather than at boot: a
  // GenericRenderWindow in a display:none container measures 0x0 and would sit
  // there rendering to a zero-size canvas for a layout the user may never open.
  function ensurePanes() {
    if (views.K) return;
    views.K = makeView("vK", SM.K, document.getElementById("iK"));
    views.J = makeView("vJ", SM.J, document.getElementById("iJ"));
    views.I = makeView("vI", SM.I, document.getElementById("iI"));
    if (curImg) KEYS.forEach(function (k) {
      views[k].mapper.setInputData(curImg.ct);
      views[k].segMapper.setInputData(curImg.sg);
      orientCamera(views[k]);
    });
    applyWL();
    applyVis();
    KEYS.forEach(function (k) { setSlice(k, +document.getElementById("sl" + k).value); });
  }

  function setLayout(mode) {
    document.getElementById("views").className = mode;
    document.getElementById("layout").value = mode;
    if (history.replaceState) history.replaceState(null, "", "#" + mode);
    document.getElementById("planeGrp").style.display = mode === "mode3d" ? "" : "none";
    // Size is read from the container, so a pane coming back into view has to
    // resize after the class change has taken effect, not before it.
    requestAnimationFrame(function () {
      if (mode === "panes") ensurePanes();
      KEYS.forEach(function (k) { if (views[k]) views[k].grw.resize(); });
      if (v3) v3.grw.resize();
      renderAll();
    });
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    if (typeof vtk === "undefined") {
      status("vtk.js failed to load — rebuild the viewer with --vtk pointing at " +
             "the UMD bundle", true);
      return;
    }
    SM = vtk.Rendering.Core.vtkImageMapper.SlicingMode;
    v3 = make3D("v3");
    setLayout(location.hash === "#panes" ? "panes" : "mode3d");

    KEYS.forEach(function (k) {
      document.getElementById("sl" + k).addEventListener("input", function (e) {
        setSlice(k, +e.target.value);
      });
      document.getElementById("p" + k).addEventListener("change", applyVis);
    });
    ["win", "lev"].forEach(function (id) {
      document.getElementById(id).addEventListener("input", applyWL);
    });
    document.getElementById("seg").addEventListener("change", applyVis);
    document.getElementById("home3d").addEventListener("click", function () {
      orient3D(true);
    });
    document.getElementById("layout").addEventListener("change", function (e) {
      setLayout(e.target.value);
    });
    document.getElementById("colorBy").addEventListener("change", function () {
      draw2(); draw3();
    });

    c2.addEventListener("click", function (e) {
      var i = nearest(pts2, e, c2); if (i >= 0) select(i);
    });
    var drag = null;
    c3.addEventListener("mousedown", function (e) { drag = { x: e.clientX, y: e.clientY, moved: 0 }; });
    window.addEventListener("mousemove", function (e) {
      if (!drag) return;
      yaw += (e.clientX - drag.x) * 0.01;
      pitch += (e.clientY - drag.y) * 0.01;
      pitch = Math.max(-1.5, Math.min(1.5, pitch));
      drag.moved += Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y);
      drag.x = e.clientX; drag.y = e.clientY;
      draw3();
    });
    window.addEventListener("mouseup", function (e) {
      if (drag && drag.moved < 4) { var i = nearest(pts3, e, c3); if (i >= 0) select(i); }
      drag = null;
    });
    window.addEventListener("resize", function () {
      draw2(); draw3();
      KEYS.forEach(function (k) { if (views[k]) views[k].grw.resize(); });
      if (v3) v3.grw.resize();
    });

    fetch("data/embedding.json").then(function (r) { return r.json(); }).then(function (j) {
      data = j;
      if (j.dims) DIMS = j.dims;
      if (j.spacing) SPACING = j.spacing;
      // Each pane scrubs its own axis (K=z, J=y, I=x), so its range and start
      // slice come from that axis. Sharing DIMS[0] let the short axes scrub
      // past the end of the volume.
      [["K", 2], ["J", 1], ["I", 0]].forEach(function (kv) {
        var el = document.getElementById("sl" + kv[0]);
        el.max = DIMS[kv[1]] - 1;
        el.value = Math.floor((DIMS[kv[1]] - 1) / 2);
      });
      draw2(); draw3();
      var nreal = j.cases.filter(function (c) { return c.real; }).length;
      status(j.cases.length + " cases (" + nreal + " real, " +
        (j.cases.length - nreal) + " synthetic) — click a point");
      select(0);
    }).catch(function (e) { status("failed to load embedding.json: " + e.message, true); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
