module.exports = {
  extends: ['stylelint-config-standard', 'stylelint-config-recess-order', 'stylelint-config-prettier'],
  plugins: ['stylelint-scss'],
  rules: {
    // Limit the number of universal selectors in a selector to avoid very slow selectors
    'selector-max-universal': 1,
    'at-rule-no-unknown': null,
    'unicode-bom': 'never',
    'scss/at-rule-no-unknown': true,
    'selector-pseudo-class-no-unknown': [
      true,
      {
        ignorePseudoClasses: ['export'],
      },
    ],
  },
};
