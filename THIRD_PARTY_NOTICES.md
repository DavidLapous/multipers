# Third-Party Notices

This distribution includes and/or links against third-party components with GPL/LGPL licenses.
As a result, distributed `multipers` artifacts built with these components are licensed under
`GPL-3.0-or-later`.

## Components

- `AIDA`
  - Upstream: https://github.com/JanJend/AIDA
  - Source in this repository: `ext/AIDA/`
  - Commit used in this workspace: `d820f64984323f2886d0a997eb247dd7acfcd0c7`
  - License: GPL-3.0-or-later (see `ext/AIDA/LICENSE`)

- `Persistence-Algebra`
  - Upstream: https://github.com/JanJend/Persistence-Algebra
  - Source in this repository: `ext/Persistence-Algebra/`
  - Commit used in this workspace: `381ed521b9ea427a996da4d8d9b788a734cac628`
  - License: GPL-3.0-or-later (see `ext/Persistence-Algebra/LICENSE`)

- `function_delaunay`
  - Upstream: https://bitbucket.org/mkerber/function_delaunay/
  - Header source path used at build time: `ext/function_delaunay`
  - Commit used in this workspace: `a481f8d90af7a97795c467d81fb3db2cf952a58e`
  - License: GPL-3.0-or-later (see upstream `COPYING`)

- `rhomboidtiling_newer_cgal_version`
  - Upstream: https://github.com/DavidLapous/rhomboidtiling_newer_cgal_version
  - Header/source path used at build time: `ext/rhomboidtiling_newer_cgal_version`
  - Commit used in this workspace: `c414cdc60f30196ffd3ce5cda1817368655905f0`
  - License: MIT (see `ext/rhomboidtiling_newer_cgal_version/LICENSE.md`)

- `mpfree`
  - Upstream: https://bitbucket.org/mkerber/mpfree/
  - Header source path used at build time: `ext/mpfree`
  - Commit used in this workspace: `8a423f7e7997b744d89405ee579b4e2f1679bbb2`
  - License: LGPL-3.0-or-later (see upstream `COPYING.LESSER`)

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
