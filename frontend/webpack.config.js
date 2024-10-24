const path = require('path');
const webpack = require('webpack');

module.exports = {
    entry: './src/main.js', // أو main.js إذا كنت تستخدمه كملف الدخول
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
    mode: 'development',
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env'],
                    },
                },
            },
        ],
    },
    resolve: {
        fallback: {
            fs: false,  // تجاهل وحدة fs
            crypto: false,  // تجاهل وحدة crypto
            util: false,  // تجاهل وحدة util
        },
    },
    plugins: [
        new webpack.IgnorePlugin({
            resourceRegExp: /^node-fetch$/, // تجاهل node-fetch
        }),
    ],
};
