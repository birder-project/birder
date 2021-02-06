module.exports = {
  lintOnSave: false,
  outputDir: './dist/',
  devServer: {
    port: 8888,
    historyApiFallback: true,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': '*',
      'Access-Control-Allow-Headers': '*',
      'Access-Control-Allow-Credentials': 'true',
    },
  },
};
