{pkgs, ...}: {
  languages.clojure.enable = true;
  languages.java.enable = true;

  packages = with pkgs; [
    git
    clj-kondo
    cljfmt
    clojure-lsp
    babashka
  ];

  enterShell = ''
    echo "Checking Clojure developer tools..."

    echo "🚀 Functional Clojure environment ready!"
    echo "💡 Available commands: fmt, lint, test (or run-tests), check, run"
  '';

  scripts = {
    fmt.exec = ''
      if [ $# -eq 0 ]; then
        echo "Formatting all .clj, .cljs, .cljc, .edn files..."
        cljfmt fix src test deps.edn
      else
        cljfmt fix "$@"
      fi
    '';

    lint.exec = ''
      if [ $# -eq 0 ]; then
        echo "Linting all source and test files..."
        clj-kondo --lint src test deps.edn
      else
        clj-kondo --lint "$@"
      fi
    '';

    test.exec = ''
      echo "Running Clojure tests..."
      if [ $# -eq 0 ]; then
        clojure -M:test
      else
        clojure -M:test "$@"
      fi
    '';

    run-tests.exec = ''
      echo "Running Clojure tests..."
      if [ $# -eq 0 ]; then
        clojure -M:test
      else
        clojure -M:test "$@"
      fi
    '';

    run.exec = ''
      clojure -M:run "$@"
    '';

    check.exec = ''
      set -e
      echo "=== 1. Formatting ==="
      cljfmt check src test deps.edn
      echo ""
      echo "=== 2. Linting ==="
      clj-kondo --lint src test deps.edn
      echo ""
      echo "=== 3. Testing ==="
      run-tests "$@"
      echo ""
      echo "✅ All checks passed successfully!"
    '';
  };

  pre-commit.hooks = {
    cljfmt = {
      enable = true;
      name = "Clojure Formatter";
      entry = "cljfmt fix";
      files = "\\.(clj|cljs|cljc|edn)$";
      language = "system";
      pass_filenames = true;
    };
    clj-kondo = {
      enable = true;
      name = "Clojure Linter";
      entry = "clj-kondo --lint";
      files = "\\.(clj|cljs|cljc|edn)$";
      language = "system";
      pass_filenames = true;
    };
  };
}
