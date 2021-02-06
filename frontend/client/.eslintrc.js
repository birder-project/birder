module.exports = {
  root: true,

  env: {
    browser: true,
    node: true,
  },

  extends: ['plugin:vue/essential', '@vue/airbnb', 'plugin:no-unsanitized/DOM'],

  parserOptions: {
    parser: 'babel-eslint',
  },

  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
    'max-len': ['error', 110],
    'arrow-parens': ['error', 'always'],
    'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    'operator-linebreak': ['error', 'after'],
    'no-useless-constructor': 'off',
    'no-empty-function': 'off',
    'object-curly-newline': 'off',
  },

  overrides: [
    {
      files: ['**/__tests__/*.{j,t}s?(x)', '**/tests/unit/**/*.spec.{j,t}s?(x)'],
      env: {
        jest: true,
      },
    },
  ],
};
