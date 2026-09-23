"""python -m app <команда> — то же, что python -m app.cli."""
from app.cli import main

raise SystemExit(main())
