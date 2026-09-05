/* morphome synthetic corpus explorer.
 *
 * Left: cascaded-UMAP scatters (2D clickable, 3D drag-to-rotate), real and
 * synthetic co-embedded. Right: vtk.js slices of the selected case, loaded on
 * demand from data/vol/<case>_{ct,seg}.bin, in either of two layouts --
 *
 *   mode3d  the three planes as intersecting slices in one rotatable 3D scene
 *   panes   the three planes as separate parallel-projection views
 *
 * -- sharing one set of slice indices, window/level and organ overlay. In the
 * 3D layout each structure can additionally be shown as a marching-cubes
 * surface, contoured in the browser (see "surfaces" below).
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

  // Opacity given to the air outside the patient. Not zero: a faint surround
  // still shows where the slice plane physically is, which matters in the 3D
  // layout where an invisible plane would leave its two neighbours floating.
  var AIR_ALPHA = 0.05;

  // Label value -> colour, index 0 being background. Shared by the slice
  // overlay and the 3D surfaces so a structure is the same colour however it
  // is drawn; a surface that did not match its own overlay would be worse than
  // no surface at all.
  var LABEL_RGB = [[0, 0, 0], [0.90, 0.10, 0.10], [1.00, 0.95, 0.20],
    [0.20, 0.60, 1.00], [0.20, 1.00, 0.40], [0.00, 0.70, 0.25],
    [1.00, 0.45, 0.00], [0.85, 0.30, 0.00], [0.80, 0.30, 1.00],
    [0.55, 0.10, 0.85]];

  function rgbCss(i) {
    var c = LABEL_RGB[i] || [0.6, 0.6, 0.6];
    return "rgb(" + Math.round(c[0] * 255) + "," + Math.round(c[1] * 255) +
      "," + Math.round(c[2] * 255) + ")";
  }

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
  // Optional: the 2D map is not in the page by default, leaving the 3D one as
  // the selector. Everything here no-ops when the canvas is absent rather than
  // being deleted, so putting the markup back is enough to bring it and its
  // click-to-select behaviour back.
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
    if (!c2) return;
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
      if (!p) return;
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
  var zoom3 = 1, panX = 0, panY = 0;

  // Each synthetic case can draw as a billboard of its own anatomy instead of
  // a dot, from the atlas render_thumbs.py bakes. 400 sprites do not fit in
  // this canvas at once, so which cases get one is decided per frame: nearest
  // first, and only where the sprite would not collide with one already
  // placed. Everything else stays a dot. Zooming in spreads the cloud, more
  // sprites clear the test, and the anatomy appears progressively -- so the
  // map is readable at every zoom instead of legible at exactly one.
  var thumbs = null, thumbImg = null, thumbIndex = {};

  function loadThumbs() {
    fetch("data/thumbs.json").then(function (r) {
      if (!r.ok) throw new Error("no thumbnail atlas");
      return r.json();
    }).then(function (j) {
      var img = new Image();
      img.onload = function () { thumbs = j; thumbImg = img; draw3(); };
      img.onerror = function () { console.warn("thumbs.png failed to load"); };
      j.order.forEach(function (name, i) { thumbIndex[name] = i; });
      img.src = "data/thumbs.png";
    }).catch(function () {
      // Optional asset: without it the map is the scatter it always was.
      console.info("no thumbnail atlas; run scripts/render_thumbs.py");
    });
  }

  function thumbSize() {
    return +document.getElementById("thumbSz").value * Math.min(2.2, Math.sqrt(zoom3));
  }

  function thumbsOn() {
    return thumbImg && document.getElementById("thumbOn").checked;
  }

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
    var s = (Math.min(f.w, f.h) / 2 - pad) * zoom3;
    // A hidden case gets no screen point at all, so it drops out of hit
    // testing along with the drawing -- otherwise clicking empty space could
    // still select a real case that is not on screen.
    var withReal = document.getElementById("showReal").checked;
    pts3 = proj.map(function (p, i) {
      if (!withReal && data.cases[i].real) return null;
      return { x: f.w / 2 + p.x * s + panX, y: f.h / 2 - p.y * s + panY };
    });
    var order = proj.map(function (p, i) { return i; })
      .sort(function (a, b) { return proj[a].d - proj[b].d; });

    // Claim sprite slots nearest-first, so when two cases compete for the same
    // patch of canvas the one in front wins -- which is also the one whose
    // sprite would be drawn on top anyway.
    var sprite = {};
    if (thumbsOn()) {
      var size = thumbSize(), sep = size * 0.8, taken = [];
      for (var k = order.length - 1; k >= 0; k--) {
        var i3 = order[k];
        if (data.cases[i3].real) continue;              // synthetic only
        if (!(data.cases[i3].name in thumbIndex)) continue;
        var p3 = pts3[i3];
        if (!p3) continue;
        if (p3.x < -size || p3.y < -size || p3.x > f.w + size || p3.y > f.h + size)
          continue;                                     // off-canvas
        var clash = false;
        for (var t3 = 0; t3 < taken.length; t3++) {
          if (Math.abs(taken[t3].x - p3.x) < sep && Math.abs(taken[t3].y - p3.y) < sep) {
            clash = true; break;
          }
        }
        if (!clash) { taken.push(p3); sprite[i3] = true; }
      }
    }

    order.forEach(function (i) {
      var rec = data.cases[i], p = pts3[i];
      if (!p) return;
      var t = (proj[i].d + 1.8) / 3.6;
      if (sprite[i]) {
        var sz = thumbSize() * (0.72 + 0.45 * t);
        var n = thumbIndex[rec.name], cell = thumbs.cell;
        ctx.globalAlpha = 0.55 + 0.45 * t;
        ctx.drawImage(thumbImg,
          (n % thumbs.cols) * cell, Math.floor(n / thumbs.cols) * cell, cell, cell,
          p.x - sz / 2, p.y - sz / 2, sz, sz);
        if (i === sel) {
          ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = "#fff";
          ctx.strokeRect(p.x - sz / 2, p.y - sz / 2, sz, sz);
        }
        return;
      }
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
      if (!p) return;                                   // hidden -> not selectable
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

    // Air outside the patient is made nearly transparent so the surfaces and
    // the other two planes show through the surround instead of being buried
    // in it.
    //
    // The cut needs no threshold to tune: the CT is body-masked at preprocess
    // time, so every voxel outside the patient sits exactly on the air floor
    // and nothing inside does. Measured over the corpus, the fraction of
    // voxels equal to 0 matches 1 - body_frac to three decimals -- 0.778 vs
    // 0.778 on LUNG1-001, 0.703 vs 0.703 on synth0000 -- so 0 *is* the
    // outside, and the first occupied value above it is already anatomy.
    //
    // The ramp is anchored to the piecewise function's own range rather than
    // to the window: the image mapper derives pwfshift/pwfscale from
    // pwf.getRange(), and only the colour uniforms come from
    // colorWindow/colorLevel. So the window and level sliders stay a pure
    // contrast stretch and do not drag the transparency cut around with them.
    var ctOpacity = vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();
    ctOpacity.addPoint(0, AIR_ALPHA);
    ctOpacity.addPoint(1, 1.0);
    ctOpacity.addPoint(255, 1.0);
    var ap = actor.getProperty();
    if (ap.setPiecewiseFunction) ap.setPiecewiseFunction(0, ctOpacity);
    else if (ap.setScalarOpacity) ap.setScalarOpacity(0, ctOpacity);

    // A per-voxel opacity function is not enough on its own. An image slice is
    // classified by `getIsOpaque()`, which looks at the *actor* opacity and
    // ignores the piecewise function entirely, so a plane with transparent air
    // still counts as opaque -- and this bundle gives image props an
    // `opaquePass` that does `depthMask(true)` with no translucent pass taken.
    // The plane then writes depth across its whole quad, air included, and
    // whichever plane is drawn first hides everything behind it. That is why
    // the transverse plane in particular looked solid: it is added first, so it
    // claimed the depth buffer for the other two and for the surfaces.
    //
    // forceTranslucent flips the classification, which routes the plane to the
    // translucent pass where `depthMask(false)` applies. The air then really
    // shows what is behind it. The surfaces are opaque and drawn first, so they
    // still occlude the planes correctly; what is given up is depth ordering
    // *between* the three planes, which no longer write depth against each
    // other and so blend in the order they were added.
    if (actor.setForceTranslucent) actor.setForceTranslucent(true);

    ren.addActor(actor);

    var segMapper = C.vtkImageMapper.newInstance();
    segMapper.setSlicingMode(mode);
    var segActor = C.vtkImageSlice.newInstance();
    segActor.setMapper(segMapper);

    // 0 stays transparent so background never paints over the CT.
    var ctf = C.vtkColorTransferFunction.newInstance();
    LABEL_RGB.forEach(function (c, i) { ctf.addRGBPoint(i, c[0], c[1], c[2]); });
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
    // Same reasoning as the CT plane: the overlay is transparent at label 0 via
    // its piecewise function, which the opaque/translucent classification does
    // not look at, so without this it would write depth over the whole quad too.
    if (segActor.setForceTranslucent) segActor.setForceTranslucent(true);

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

  // Plain trackball rotation is free-form: drag sideways twice and the horizon
  // has rolled, so superior is no longer up and the orientation cube is the
  // only thing left saying so. The manipulator style's rotate takes a fixed
  // world up instead -- azimuth turns about it, elevation is clamped short of
  // the poles, and the view-up is re-applied after every drag -- so up stays
  // up however far the camera is dragged. +z is superior in LPS, which is the
  // same view-up orient3D homes to, so the reset button and the interaction
  // agree.
  //
  // Falling back to trackball rather than failing keeps a bundle without the
  // manipulators module usable, just with the old free rotation.
  function stableStyle(iren) {
    var S = vtk.Interaction.Style, M = vtk.Interaction.Manipulators;
    if (!M || !S.vtkInteractorStyleManipulator) {
      iren.setInteractorStyle(S.vtkInteractorStyleTrackballCamera.newInstance());
      return null;
    }
    var style = S.vtkInteractorStyleManipulator.newInstance();
    style.addMouseManipulator(M.vtkMouseCameraTrackballRotateManipulator.newInstance({
      button: 1, useWorldUpVec: true, worldUpVec: [0, 0, 1] }));
    style.addMouseManipulator(M.vtkMouseCameraTrackballPanManipulator.newInstance({
      button: 1, shift: true }));
    style.addMouseManipulator(M.vtkMouseCameraTrackballPanManipulator.newInstance({
      button: 2 }));
    style.addMouseManipulator(M.vtkMouseCameraTrackballZoomManipulator.newInstance({
      button: 3 }));
    // Unlike the trackball style, this one has no wheel binding of its own.
    style.addMouseManipulator(M.vtkMouseCameraTrackballZoomManipulator.newInstance({
      scrollEnabled: true, dragEnabled: false }));
    if (M.vtkGestureCameraManipulator && style.addGestureManipulator)
      style.addGestureManipulator(M.vtkGestureCameraManipulator.newInstance());
    iren.setInteractorStyle(style);
    return style;
  }

  // The style pushes its centerOfRotation onto the manipulator on every button
  // down, and it defaults to [0, 0, 0]. The volume's origin is 0, so that
  // point is a *corner* of it -- on the thorax frame the corner of a
  // 672 x 480 x 288 mm box -- and the planes swing around it on a long arm
  // instead of turning in place. Anchor it to the middle of the grid.
  function setRotationCenter() {
    if (!v3 || !v3.style || !v3.style.setCenterOfRotation) return;
    v3.style.setCenterOfRotation(
      (DIMS[0] - 1) * SPACING / 2,
      (DIMS[1] - 1) * SPACING / 2,
      (DIMS[2] - 1) * SPACING / 2);
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
    var style = stableStyle(iren);

    var planes = {};
    [["K", 2], ["J", 1], ["I", 0]].forEach(function (kv) {
      planes[kv[0]] = slicePair(ren, kv[1]);
    });
    ren.getActiveCamera().setParallelProjection(false);
    addCube(ren, iren);
    return { grw: grw, ren: ren, rw: rw, planes: planes, style: style };
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

  // -------------------------------------------------------------- surfaces
  // Surfaces do NOT read the seg volume. data/vol/<case>_seg.bin is
  // winner-take-all -- one byte per voxel, earlier structure index wins --
  // which is the right representation for tinting a slice and lossy exactly
  // where contours overlap. On thorax that erases the two structures most
  // worth seeing in 3D: Lungs is the union of Lung_L/Lung_R and loses every
  // voxel to them, while GTV is the last index sitting inside a lung and keeps
  // as little as none of itself. Contouring that map would render an empty
  // tumour.
  //
  // So they read data/mask/<case>.bin: per-structure bitmasks, each cropped to
  // its own bounding box, written by scripts/export_masks.py straight from the
  // cache's label bitplanes with no overlap rule applied. The format is
  // documented there. Cropping is also what makes marching cubes affordable in
  // the browser -- a GTV box is ~70x smaller than the full 224x160x96 grid, so
  // the filter runs over what the structure actually occupies.
  var SURF_SMOOTH = 2;                     // [1,2,1] passes per axis
  var HAS_MC = false;                      // bundle ships marching cubes
  var MASK_LRU = 16, POLY_LRU = 24;
  var maskCache = {}, maskOrder = [];      // case -> {buf, recs}
  var polyCache = {}, polyOrder = [];      // "case|i" -> vtkPolyData
  var surfActors = {};                     // structure index -> {actor, mapper}
  var surfOn = {};                         // structure name -> shown, once chosen
  var surfGen = 0;                         // abandons builds for a stale case
  var restStatus = "";                     // status to return to when a build ends
  var maskMeta = null;                     // case -> {present, redundant}

  // Until the reader picks structures for themselves, the shown set is derived
  // per case: everything contoured, minus the union channels. That is what
  // makes a default of "all of them" usable -- drawing Lungs on top of
  // Lung_L/Lung_R would put two coincident surfaces in the same place and bury
  // the tumour inside them. The rule lives in export_masks.py and arrives as
  // data/mask_meta.json, so the browser and the offline thumbnails cannot
  // disagree about it.
  var surfAuto = true;

  function autoSet(name) {
    var mm = maskMeta && maskMeta[name];
    if (!mm) return [];
    return mm.present.filter(function (i) { return mm.redundant.indexOf(i) < 0; });
  }

  // The structures to draw for a case, honouring an explicit choice if one has
  // been made and falling back to the derived default if not.
  function wanted(name) {
    if (surfAuto) return autoSet(name);
    var out = [];
    data.structures.forEach(function (s, i) { if (surfOn[s]) out.push(i); });
    return out;
  }

  // Freeze the current derived set into an explicit one, so the first click on
  // a checkbox toggles that structure rather than silently turning off every
  // other structure the default had switched on.
  function pinChoice(name) {
    if (!surfAuto) return;
    surfAuto = false;
    surfOn = {};
    autoSet(name).forEach(function (i) { surfOn[data.structures[i]] = true; });
  }

  function lru(cache, order, key, cap) {
    var at = order.indexOf(key);
    if (at >= 0) order.splice(at, 1);
    order.push(key);
    while (order.length > cap) delete cache[order.shift()];
  }

  function parseMasks(buf) {
    var dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x4b534d4d)      // "MMSK"
      throw new Error("not a mask file");
    if (dv.getUint8(4) !== 1)
      throw new Error("mask format v" + dv.getUint8(4) + " is not supported");
    var recs = [];
    for (var i = 0, n = dv.getUint8(5); i < n; i++) {
      var p = 8 + 20 * i;
      recs.push({
        x0: dv.getUint16(p, true), y0: dv.getUint16(p + 2, true),
        z0: dv.getUint16(p + 4, true), nx: dv.getUint16(p + 6, true),
        ny: dv.getUint16(p + 8, true), nz: dv.getUint16(p + 10, true),
        off: dv.getUint32(p + 12, true), len: dv.getUint32(p + 16, true)
      });
    }
    return { buf: buf, recs: recs };
  }

  // Loaded before the first case is selected, since it decides which surfaces
  // are on by default. A viewer built before this file existed simply gets no
  // surfaces until the reader ticks one.
  function loadMaskMeta() {
    return fetch("data/mask_meta.json").then(function (r) {
      if (!r.ok) throw new Error("absent");
      return r.json();
    }).then(function (j) { maskMeta = j.cases; }).catch(function () {
      console.info("no mask_meta.json; run scripts/export_masks.py --meta-only");
    });
  }

  function fetchMasks(name) {
    if (maskCache[name]) {
      lru(maskCache, maskOrder, name, MASK_LRU);
      return Promise.resolve(maskCache[name]);
    }
    return fetch("data/mask/" + name + ".bin").then(function (r) {
      if (!r.ok) throw new Error(r.status + " — run scripts/export_masks.py");
      return r.arrayBuffer();
    }).then(function (buf) {
      var mk = parseMasks(buf);
      maskCache[name] = mk;
      lru(maskCache, maskOrder, name, MASK_LRU);
      return mk;
    });
  }

  // Bits -> a scalar field, padded by one zero voxel on every face so a
  // structure flush against its own bounding box still closes instead of
  // contouring to an open shell.
  function unpackField(buf, rec) {
    var nx = rec.nx + 2, ny = rec.ny + 2, nz = rec.nz + 2;
    var f = new Float32Array(nx * ny * nz);
    var bits = new Uint8Array(buf, rec.off, rec.len);
    var i = 0;
    for (var z = 0; z < rec.nz; z++) {
      for (var y = 0; y < rec.ny; y++) {
        var base = ((z + 1) * ny + (y + 1)) * nx + 1;
        for (var x = 0; x < rec.nx; x++, i++) {
          if ((bits[i >> 3] >> (i & 7)) & 1) f[base + x] = 1;
        }
      }
    }
    return { f: f, nx: nx, ny: ny, nz: nz };
  }

  // Separable [1,2,1]/4. A binary mask at 3 mm contours to a visible voxel
  // staircase; blurring the field first and taking the 0.5 isosurface removes
  // it. 0.5 is the kernel's symmetric midpoint, so the surface does not
  // migrate inward or outward -- the enclosed volume is preserved, which
  // matters when the point of the viewer is comparing synthetic anatomy to
  // real.
  function smooth(g, passes) {
    var nx = g.nx, ny = g.ny, nz = g.nz, n = nx * ny * nz;
    var src = g.f, dst = new Float32Array(n), swap;
    var strides = [1, nx, nx * ny], counts = [nx, ny, nz];
    for (var p = 0; p < passes; p++) {
      for (var a = 0; a < 3; a++) {
        var s = strides[a], m = counts[a];
        for (var i = 0; i < n; i++) {
          var k = Math.floor(i / s) % m, c = src[i];
          dst[i] = 0.25 * ((k > 0 ? src[i - s] : c) + 2 * c +
                           (k < m - 1 ? src[i + s] : c));
        }
        swap = src; src = dst; dst = swap;
      }
    }
    g.f = src;
    return g;
  }

  // The crop sits at its true place in the volume: the padded field starts one
  // voxel before the bounding box, and the slice planes use origin 0 with the
  // same spacing, so surface and slices land in the same world space with no
  // registration step.
  function contour(g, rec) {
    var img = vtk.Common.DataModel.vtkImageData.newInstance();
    img.setDimensions(g.nx, g.ny, g.nz);
    img.setSpacing(SPACING, SPACING, SPACING);
    img.setOrigin((rec.x0 - 1) * SPACING, (rec.y0 - 1) * SPACING,
                  (rec.z0 - 1) * SPACING);
    img.getPointData().setScalars(vtk.Common.Core.vtkDataArray.newInstance({
      name: "mask", numberOfComponents: 1, values: g.f }));
    var mc = vtk.Filters.General.vtkImageMarchingCubes.newInstance({
      contourValue: 0.5, computeNormals: true, mergePoints: true });
    mc.setInputData(img);
    return mc.getOutputData();
  }

  // Surfaces default to fully opaque, and that is load-bearing rather than
  // cosmetic. This bundle has no depth peeling -- `useDepthPeeling` exists on
  // the renderer as a property with no implementation behind it -- and an
  // actor counts as opaque only when `property.getOpacity() >= 1`. Only opaque
  // actors write depth. Since vtkImageSlice reports no translucent polygonal
  // geometry and the slices are drawn in their own later pass, a translucent
  // surface leaves nothing in the depth buffer for that pass to test against,
  // so the planes blend straight over it and read as floating in front of
  // anatomy they are physically behind.
  //
  // At opacity 1 the surfaces write depth, a plane behind one fails the depth
  // test and a plane in front blends over it -- both correct. Lowering the
  // slider brings the artifact back, which is what its tooltip says; to see
  // past a structure, switch it off rather than fade it.
  function surfOpacity() {
    return +document.getElementById("surfOp").value / 100;
  }

  function surfActor(i) {
    if (surfActors[i]) return surfActors[i];
    var C = vtk.Rendering.Core;
    var mapper = C.vtkMapper.newInstance({ scalarVisibility: false });
    var actor = C.vtkActor.newInstance();
    actor.setMapper(mapper);
    // Label value is index + 1, the same offset seg_uint8 writes, so a surface
    // and its slice overlay are the same colour.
    var c = LABEL_RGB[i + 1] || [0.6, 0.6, 0.6];
    actor.getProperty().setColor(c[0], c[1], c[2]);
    actor.getProperty().setOpacity(surfOpacity());
    v3.ren.addActor(actor);
    surfActors[i] = { actor: actor, mapper: mapper };
    return surfActors[i];
  }

  function applySurfOpacity() {
    var o = surfOpacity();
    Object.keys(surfActors).forEach(function (k) {
      surfActors[k].actor.getProperty().setOpacity(o);
    });
    if (v3) v3.rw.render();
  }

  // Built one structure per tick rather than in one go: a lung is ~1 M voxels
  // through blur plus marching cubes, and doing all of them synchronously
  // would freeze the page with no indication of why.
  function rebuildSurfaces() {
    if (!v3 || sel < 0 || !data || !HAS_MC) return;
    var name = data.cases[sel].name;
    var want = wanted(name);
    data.structures.forEach(function (s, i) {
      if (want.indexOf(i) < 0 && surfActors[i]) surfActors[i].actor.setVisibility(false);
    });
    var gen = ++surfGen;
    if (!want.length) { if (v3) v3.rw.render(); return; }

    fetchMasks(name).then(function (mk) {
      if (gen !== surfGen) return;
      var queue = want.slice();
      (function step() {
        if (gen !== surfGen) return;
        if (!queue.length) { status(restStatus); v3.rw.render(); return; }
        var i = queue.shift(), rec = mk.recs[i], s = data.structures[i];
        var sa = surfActor(i);
        if (!rec || !rec.len) {           // nobody contoured it in this case
          sa.actor.setVisibility(false);
          return step();
        }
        var key = name + "|" + i;
        if (polyCache[key]) {
          lru(polyCache, polyOrder, key, POLY_LRU);
        } else {
          status("building " + s + " surface …");
          polyCache[key] = contour(smooth(unpackField(mk.buf, rec), SURF_SMOOTH), rec);
          lru(polyCache, polyOrder, key, POLY_LRU);
        }
        sa.mapper.setInputData(polyCache[key]);
        sa.actor.setVisibility(true);
        v3.rw.render();
        setTimeout(step, 0);
      })();
    }).catch(function (e) {
      if (gen === surfGen) status("failed to load masks: " + e.message, true);
    });
  }

  function select(i) {
    if (i < 0 || i === sel) return;
    sel = i;
    var rec = data.cases[i];
    showMeta(rec);
    draw2(); draw3();
    // Drop the previous case's surfaces now rather than when the new ones are
    // ready: they belong to different anatomy, and leaving them up while the
    // slices change underneath is actively misleading.
    surfGen++;
    Object.keys(surfActors).forEach(function (k) {
      surfActors[k].actor.setVisibility(false);
    });
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
        restStatus = rec.name + (rec.real ? "  (real case)" : "  (synthetic)");
        status(restStatus);
        rebuildSurfaces();
      })
      .catch(function (e) { status("failed to load volume: " + e.message, true); });
  }

  // Surface toggles, one per structure, in the control bar. They were in the
  // side panel next to the voxel counts, which reads well until the window is
  // short: the panel is the first thing the layout squeezes, and the toggles
  // ended up in a sliver nobody could use. The counts stay in the panel, where
  // being cut off costs only a number; the controls sit in the bar, which is
  // always the same height.
  //
  // A structure nobody contoured in this case has nothing to contour, so its
  // toggle is disabled rather than offering a click that does nothing.
  function showStructToggles(rec) {
    var host = document.getElementById("structList");
    var want = wanted(rec.name);
    var mm = maskMeta && maskMeta[rec.name];
    var html = "";
    data.structures.forEach(function (s, i) {
      var vox = rec.organ_voxels[s] || 0;
      var union = mm && mm.redundant.indexOf(i) >= 0;
      html += "<label class='ck" + (vox ? "" : " off") + "' title='" + s +
        (vox ? " — " + vox.toLocaleString() + " voxels" : " — not contoured") +
        (union ? " — union of smaller structures, off by default" : "") + "'>" +
        "<input type='checkbox' data-i='" + i + "'" +
        (want.indexOf(i) >= 0 ? " checked" : "") + (vox ? "" : " disabled") + ">" +
        "<i class='dot' style='background:" + rgbCss(i + 1) + "'></i>" + s + "</label>";
    });
    host.innerHTML = html;
    Array.prototype.forEach.call(host.querySelectorAll("input"), function (el) {
      el.addEventListener("change", function () {
        pinChoice(rec.name);
        surfOn[data.structures[+el.getAttribute("data-i")]] = el.checked;
        rebuildSurfaces();
      });
    });
  }

  function showMeta(rec) {
    var rows = [
      ["case", rec.name], ["corpus", rec.real ? "real" : "synthetic"],
      ["organs present", rec.n_present + " / " + data.structures.length],
      ["bone sharpness", (rec.sharpness || 0).toFixed(1)],
      ["body fraction", (rec.body_frac || 0).toFixed(3)]
    ];
    var html = "<h2>Selected case</h2><table>";
    rows.forEach(function (r) { html += "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td></tr>"; });
    html += "</table><h2>Structure voxels</h2><table>";
    data.structures.forEach(function (s, i) {
      html += "<tr><td><i class='dot' style='background:" + rgbCss(i + 1) +
        "'></i>" + s + "</td><td>" +
        (rec.organ_voxels[s] || 0).toLocaleString() + "</td></tr>";
    });
    metaEl.innerHTML = html + "</table>";
    showStructToggles(rec);
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
    // Surfaces live only in the 3D renderer -- drawn into a parallel-projection
    // pane they would be a silhouette floating over the slice, so the control
    // goes away with the layout rather than doing nothing.
    ["planeGrp", "surfGrp"].forEach(function (id) {
      document.getElementById(id).style.display = mode === "mode3d" ? "" : "none";
    });
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
    // Not every vtk.js build ships the filters module. Slices are the point of
    // the viewer and surfaces are an addition, so a bundle without marching
    // cubes loses the surfaces and keeps everything else.
    HAS_MC = !!(vtk.Filters && vtk.Filters.General &&
                vtk.Filters.General.vtkImageMarchingCubes);
    if (!HAS_MC) {
      console.warn("vtk.js bundle has no vtkImageMarchingCubes; surfaces disabled");
      document.getElementById("surfGrp").style.display = "none";
    }
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
    document.getElementById("surfOp").addEventListener("input", applySurfOpacity);
    document.getElementById("surfNone").addEventListener("click", function () {
      // An explicit empty choice, not a return to the derived default -- which
      // would switch everything straight back on.
      surfAuto = false;
      surfOn = {};
      if (sel >= 0) showStructToggles(data.cases[sel]);
      rebuildSurfaces();
    });
    document.getElementById("layout").addEventListener("change", function (e) {
      setLayout(e.target.value);
    });
    document.getElementById("colorBy").addEventListener("change", function () {
      draw2(); draw3();
    });

    if (c2) c2.addEventListener("click", function (e) {
      var i = nearest(pts2, e, c2); if (i >= 0) select(i);
    });
    var drag = null;
    c3.addEventListener("mousedown", function (e) {
      // Right button or shift pans, left rotates -- the same split the 3D
      // ortho view uses, so the two 3D things on screen behave alike.
      drag = { x: e.clientX, y: e.clientY, moved: 0,
               pan: e.button === 2 || e.shiftKey };
      if (drag.pan) e.preventDefault();
    });
    c3.addEventListener("contextmenu", function (e) { e.preventDefault(); });
    window.addEventListener("mousemove", function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (drag.pan) {
        panX += dx; panY += dy;
      } else {
        yaw += dx * 0.01;
        pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.01));
      }
      drag.moved += Math.abs(dx) + Math.abs(dy);
      drag.x = e.clientX; drag.y = e.clientY;
      draw3();
    });
    window.addEventListener("mouseup", function (e) {
      if (drag && drag.moved < 4) { var i = nearest(pts3, e, c3); if (i >= 0) select(i); }
      drag = null;
    });
    // Zoom about the cursor, so the case under the pointer stays under it.
    c3.addEventListener("wheel", function (e) {
      e.preventDefault();
      var r = c3.getBoundingClientRect();
      var mx = e.clientX - r.left, my = e.clientY - r.top;
      var k = Math.exp(-e.deltaY * 0.0015);
      var next = Math.max(0.5, Math.min(40, zoom3 * k));
      k = next / zoom3;
      panX = mx - (mx - panX - r.width / 2) * k - r.width / 2;
      panY = my - (my - panY - r.height / 2) * k - r.height / 2;
      zoom3 = next;
      draw3();
    }, { passive: false });
    document.getElementById("fit3").addEventListener("click", function () {
      zoom3 = 1; panX = 0; panY = 0; draw3();
    });
    document.getElementById("thumbOn").addEventListener("change", draw3);
    document.getElementById("thumbSz").addEventListener("input", draw3);
    document.getElementById("showReal").addEventListener("change", draw3);
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
      setRotationCenter();
      draw2(); draw3();
      loadThumbs();
      var nreal = j.cases.filter(function (c) { return c.real; }).length;
      status(j.cases.length + " cases (" + nreal + " real, " +
        (j.cases.length - nreal) + " synthetic) — click a point");
      // The default surface set comes out of mask_meta.json, so the first
      // selection waits for it rather than opening with nothing shown and
      // filling in a moment later.
      loadMaskMeta().then(function () { select(0); });
    }).catch(function (e) { status("failed to load embedding.json: " + e.message, true); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
