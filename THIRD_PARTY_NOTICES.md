# Third-Party Notices

This distribution includes and/or links against third-party components with GPL/LGPL licenses.
As a result, distributed `multipers` artifacts built with these components are licensed under
`GPL-3.0-or-later`.

## Components

- `AIDA`
  - Upstream: https://github.com/JanJend/AIDA
  - Source in this repository: `ext/AIDA/`
  - Commit used in this workspace: `4b5bb485a5738878783d1517ebd3b2bc0d19e13d`
  - License: GPL-3.0-or-later (see `ext/AIDA/LICENSE`)

- `Persistence-Algebra`
  - Upstream: https://github.com/JanJend/Persistence-Algebra
  - Source in this repository: `ext/Persistence-Algebra/`
  - Commit used in this workspace: `07e2c0d0bd7b6b48f6a584eee42d4c6ad583e078`
  - License: GPL-3.0-or-later (see `ext/Persistence-Algebra/LICENSE`)

- `function_delaunay`
  - Upstream: https://bitbucket.org/mkerber/function_delaunay/
  - Header source path used at build time: `ext/function_delaunay`
  - Commit used in this workspace: `f810320e4554abca7eceb25334c4940fb142f7d7`
  - License: GPL-3.0-or-later (see upstream `COPYING`)

- `deg_rips`
  - Upstream: https://bitbucket.org/mkerber/deg_rips
  - Header source path used at build time: `ext/deg_rips`
  - Commit used in this workspace: `72bd1480da902221a862f935450c9a951c3fcf8d`
  - License: LGPL-3.0-or-later (see upstream `COPYING.LESSER`)

- `rhomboidtiling_newer_cgal_version`
  - Upstream: https://github.com/DavidLapous/rhomboidtiling_newer_cgal_version
  - Header/source path used at build time: `ext/rhomboidtiling_newer_cgal_version`
  - Commit used in this workspace: `c414cdc60f30196ffd3ce5cda1817368655905f0`
  - License: MIT (see `ext/rhomboidtiling_newer_cgal_version/LICENSE.md`)

- `CGAL Spatial_searching`
  - Upstream: https://www.cgal.org/
  - License: GPL-3.0-or-later OR commercial (see SPDX headers in installed CGAL Spatial_searching package)

- `gudhi-devel`
  - Upstream: https://github.com/hschreiber/gudhi-devel
  - Header/source path used at build time: `ext/gudhi-devel`
  - Commit used in this workspace: `eeb0845c1253f66f8a54741abae6763fc1ff4245`
  - License: MIT (see `ext/gudhi-devel/LICENSE`)

- `mpfree`
  - Upstream: https://bitbucket.org/mkerber/mpfree/
  - Header source path used at build time: `ext/mpfree`
  - Commit used in this workspace: `8a423f7e7997b744d89405ee579b4e2f1679bbb2`
  - License: LGPL-3.0-or-later (see upstream `COPYING.LESSER`)

- `muphasa`
  - Upstream: https://github.com/olivergafvert/muphasa
  - Header/source path used at build time: `ext/muphasa`
  - Commit used in this workspace: `2288e91d3c4fa0d8c0307785bf565daf0faba8ab`
  - License: MIT (see `ext/muphasa/LICENSE`)

- `multi_critical`
  - Upstream: https://bitbucket.org/mkerber/multi_critical/
  - Header source path used at build time: `ext/multi_critical`
  - Commit used in this workspace: `0b41c4748c266345490510ae8b2381e9454f1fac`
  - License: LGPL-3.0-or-later (see upstream `COPYING.LESSER`)

- `multi_chunk`
  - Upstream: https://bitbucket.org/mkerber/multi_chunk/
  - Header source path used at build time: `ext/multi_chunk`
  - Commit used in this workspace: `d686f7efe623169f00951c376949b8bb30448eaf`
  - License: LGPL-3.0-or-later (see upstream `COPYING.LESSER`)

- `PHAT`
  - Upstream: https://github.com/xoltar/phat
  - Header source path used at build time: `ext/phat/include`
  - Commit used in this workspace: `872ca92b33ebe92dff2542cbdee6768e575e80aa`
  - License: LGPL-3.0-or-later (see `ext/phat/COPYING.LESSER`)

## Notes

- Build-time include paths for these dependencies are configured in `CMakeLists.txt`.
- If dependency revisions are updated, update the commit hashes listed here.
